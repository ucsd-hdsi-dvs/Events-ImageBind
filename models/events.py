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

# save events modality 
def save_events_modality_trunks():
    raise NotImplementedError

# load events modality
def load_events_modality_trunks():
    raise NotImplementedError

# preprocessor for event stream
class EventPreprocessor(RGBDTPreprocessor):
    def __init__(self,event_stem,**kwargs):
        super().__init__(rgbt_stem=event_stem,depth_stem=None,**kwargs)
    
    def forward(self, events=None):
        return super().forward(vision=events)

# initialize event model and add to image bind model
class EventModel:
    def __init__(self,
        event_embed_dim=768,
        event_kernel_size=16,
        event_num_blocks=12,
        event_num_heads=12,
        event_drop_path=0.0,
        out_embed_dim=1024):
        
        self.event_preprocessor = self.create_event_preprocessor(event_embed_dim=event_embed_dim,event_kernel_size=event_kernel_size)
        self.event_trunk = self.create_event_trunk(event_embed_dim=event_embed_dim,event_num_blocks=event_num_blocks,
                                                   event_num_heads=event_num_heads,event_drop_path=event_drop_path)
        self.event_head = self.create_event_head(out_embed_dim=out_embed_dim,event_embed_dim=event_embed_dim)
        self.event_postprocessor = self.create_event_postprocessor(out_embed_dim=out_embed_dim,logit_scale_init=10.0)
    
    def create_event_preprocessor(self,
                                  event_embed_dim=768,
                                  event_kernel_size=16):
        event_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=event_kernel_size,
                    in_channels=1,
                    out_channels=event_embed_dim,
                    stride=event_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=event_embed_dim),
        )
        event_preprocessor = EventPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            event_stem=event_stem,
        )
        return event_preprocessor
    
    def create_event_trunk(self,event_embed_dim=768,event_num_blocks=12, event_num_heads=12,event_drop_path=0.0):
        
        embed_dim=event_embed_dim
        num_blocks=event_num_blocks
        num_heads=event_num_heads
        pre_transformer_ln=False
        add_bias_kv=True
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
        
    def create_event_head(self,out_embed_dim=768,event_embed_dim=768):
        return nn.Sequential(
            nn.LayerNorm(normalized_shape=event_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(event_embed_dim, out_embed_dim, bias=False),
        )
    
    def create_event_postprocessor(self,logit_scale_init,out_embed_dim=768):
        return nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=logit_scale_init, learnable=False),
        )
    
    # add events modality preprocessor to the image bind model
    def apply_events_modality_preprocessor(self):
        return nn.ModuleDict({"event":self.event_preprocessor})

    # add events trunk to the image bind model
    def apply_events_modality_trunks(self):
        return nn.ModuleDict({"event":self.event_trunk})

    # add events modality head to the image bind model
    def apply_events_modality_head(self):
        return nn.ModuleDict({"event":self.event_head})

    # add events modality postprocessor to the image bind model
    def apply_events_modality_postprocessor(self):
        return nn.ModuleDict({"event":self.event_postprocessor})
    
    def load_weights(self,path='../.checkpoints/imagebind_huge.pth',modality="thermal"):
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
            
    
    def apply_event_layers(self,model,path='../.checkpoints/imagebind_huge.pth'):
        # check model is instance of image bind model
        assert isinstance(model,ImageBindModel)
        
        # load weights for each layers
        self.load_weights(path)
        
        # apply events modality to each layer
        model.modality_preprocessors.update(self.apply_events_modality_preprocessor())
        model.modality_trunks.update(self.apply_events_modality_trunks())
        model.modality_heads.update(self.apply_events_modality_head())
        model.modality_postprocessors.update(self.apply_events_modality_postprocessor())
        return model
    
    