import logging
import os
import math
from typing import Optional, List, Dict, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.nn.parameter import Parameter
from functools import partial

from models.helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                            SelectElement, SelectEOSAndProject)
from models.multimodal_preprocessors import (AudioPreprocessor,
                                             IMUPreprocessor, PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper,
                                             TextPreprocessor,
                                             ThermalPreprocessor)
from models.transformer import MultiheadAttention, SimpleTransformer
from models.imagebind_model import ImageBindModel
from models.imagebind_model import ModalityType

# save events modality 
def save_events_modality_trunks():
    raise NotImplementedError

# load events modality
def load_events_modality_trunks():
    raise NotImplementedError

# preprocessor for event stream
class EventPreprocessor(RGBDTPreprocessor):
    def __init__(self,event_stem,**kwargs):
        super().__init__(rgbt_stem=event_stem,**kwargs)
    
    def forward(self, event=None):
        return super().forward(vision=event)

# initialize event model and add to image bind model
class EventModel:
    def __init__(self,
        event_embed_dim=1280,
        event_kernel_size=(2, 14, 14),
        event_num_blocks=32,
        event_num_heads=16,
        event_drop_path=0.0,
        out_embed_dim=1024):
        
        self.event_preprocessor = self.create_event_preprocessor(event_embed_dim=event_embed_dim,event_kernel_size=event_kernel_size)
        self.event_trunk = self.create_event_trunk(event_embed_dim=event_embed_dim,event_num_blocks=event_num_blocks,
                                                   event_num_heads=event_num_heads,event_drop_path=event_drop_path)
        self.event_head = self.create_event_head(out_embed_dim=out_embed_dim,event_embed_dim=event_embed_dim)
        self.event_postprocessor = self.create_event_postprocessor(out_embed_dim=out_embed_dim,logit_scale_init=10.0)
    
    def create_event_preprocessor(self,
                                  event_embed_dim=1280,
                                  event_kernel_size=(2, 14, 14)):
        event_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(
                    in_channels=3,
                    kernel_size=event_kernel_size,
                    out_channels=event_embed_dim,
                    stride=event_kernel_size,
                    bias=False,
                ),
            ]
        )
        event_preprocessor = EventPreprocessor(
            event_stem=event_stem,
            img_size=[3,2,224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            depth_stem=None
        )
        return event_preprocessor
    
    def create_event_trunk(self,event_embed_dim=1280,event_num_blocks=32, event_num_heads=16,event_drop_path=0.0):
        
        embed_dim=event_embed_dim
        num_blocks=event_num_blocks
        num_heads=event_num_heads
        pre_transformer_ln=True
        add_bias_kv=False
        drop_path=event_drop_path
        
        event_trunk=SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )
        return event_trunk
        
    def create_event_head(self,out_embed_dim=1024,event_embed_dim=1280):
        return nn.Sequential(
            nn.LayerNorm(normalized_shape=event_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(event_embed_dim, out_embed_dim, bias=False),
        )
    
    def create_event_postprocessor(self,logit_scale_init,out_embed_dim=1024):
        return Normalize(dim=-1)
    
    # add events modality preprocessor to the image bind model
    def apply_events_modality_preprocessor(self):
        return nn.ModuleDict({ModalityType.EVENT:self.event_preprocessor})

    # add events trunk to the image bind model
    def apply_events_modality_trunks(self):
        return nn.ModuleDict({ModalityType.EVENT:self.event_trunk})

    # add events modality head to the image bind model
    def apply_events_modality_head(self):
        return nn.ModuleDict({ModalityType.EVENT:self.event_head})

    # add events modality postprocessor to the image bind model
    def apply_events_modality_postprocessor(self):
        return nn.ModuleDict({ModalityType.EVENT:self.event_postprocessor})
    
    def load_weights(self,path='../.checkpoints/imagebind_huge.pth',modality="vision"):
        state_dict = torch.load(path)
        preprocessor_weights=[key for key in state_dict.keys() if key.startswith(f"modality_preprocessors.{modality}")]
        trunk_weights=[key for key in state_dict.keys() if key.startswith(f"modality_trunks.{modality}")]
        head_weights=[key for key in state_dict.keys() if key.startswith(f"modality_heads.{modality}")]
        postprocessor_weights=[key for key in state_dict.keys() if key.startswith(f"modality_postprocessors.{modality}")]
        
        def load_layer(layer,weights_list):
            model_state_dict = layer.state_dict()
            
            for w in weights_list:
                w_name=w.split(".")[2:]
                model_state_dict[".".join(w_name)]=state_dict[w]
            layer.load_state_dict(model_state_dict)
        
        load_layer(self.event_preprocessor,preprocessor_weights)
        load_layer(self.event_trunk,trunk_weights)
        load_layer(self.event_head,head_weights)
        load_layer(self.event_postprocessor,postprocessor_weights)
            
    
    def apply_event_layers(self,model,path='.checkpoints/imagebind_huge.pth', load_vision_to_event=True):
        # check model is instance of image bind model
        assert isinstance(model,ImageBindModel)
        
        if load_vision_to_event:
            print('Loading vision weights to event model')
            self.load_weights(path,modality="vision")
        else:
            print('training rgblike model from scratch')
        # load weights for each layers
        # apply events modality to each layer
        model.modality_preprocessors.update(self.apply_events_modality_preprocessor())
        model.modality_trunks.update(self.apply_events_modality_trunks())
        model.modality_heads.update(self.apply_events_modality_head())
        model.modality_postprocessors.update(self.apply_events_modality_postprocessor())
        return model
    
    