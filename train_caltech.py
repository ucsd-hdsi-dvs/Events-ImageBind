# Based on PyTorch Lightning Tutorial 13 -
# SSL : https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
# Modified by Fares Abawi (@fabawi).
import logging
import os
import argparse

try:
    import comet_ml
except ImportError:
    comet_ml = None
try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("Matplotlib not installed. This is not needed if you run this script as --headless")

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import transforms

from models import imagebind_model
from models import lora as LoRA
from models.imagebind_model import ModalityType, load_module, save_module
from models.events import EventModel
from train_events import ImageBindTrain
from datasets.eclipdatasets.simple_tokenizer import tokenize

logging.basicConfig(level=logging.INFO, force=True)

# Logging settings
LOG_ON_STEP = True
LOG_ON_EPOCH = True

def load_model_from_checkpoint(checkpoint_path='/tsukimi/datasets/Chiba/finetune_checkpoint/checkpoints/imagebind-epoch=20-val_loss=0.00.ckpt'):
    """
    Load a model from a checkpoint.
    """
    imgtrain=ImageBindTrain.load_from_checkpoint(checkpoint_path,map_location=torch.device('cpu'))
    model = imgtrain.model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    return model

class CaltechTrain(L.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=1e-4, max_epochs=500, batch_size=32, num_workers=4, seed=42, 
                 self_contrast=False, temperature=0.07,  momentum_betas=(0.9, 0.95), checkpoint_path=None,class_names=None, prompt='a point cloud image of a {}'
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model=load_model_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.class_names=class_names
        self.prompt=prompt
        self.text_feats_final=None
    
    def _same_class_names(self, class_names):
        """Check if the input `class_names` matches `self.class_names`."""
        return all([c1 == c2 for c1, c2 in zip(class_names, self.class_names)])
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, 
                                betas=self.hparams.momentum_betas)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]
    
    def get_text_feats(self):
        if self.text_feats_final is not None:
            return self.text_feats_final
        
        class_names = [c.lower().replace('_', ' ') for c in class_names]
        prompts=torch.cat([tokenize(self.prompt.format(c)) for c in class_names]).to(self.device)
        
        result=[]
        for text in prompts:
            with torch.no_grad():
                text_feats = self.model({ModalityType.TEXT: text.unsqueeze(0)})
            text_feats=list(text_feats.values())[0]
            result.append(text_feats)
        result=torch.cat(result,dim=0)
        self.text_feats_final=result
        return result
    
    def get_event_feats(self, imgs):
        result=[]
        for i in range(len(imgs)):
            events=imgs[i]
            with torch.no_grad():
                event_feats = self.model({ModalityType.EVENT: events.unsqueeze(0)})
            event_feats=list(event_feats.values())[0]
            result.append(event_feats)
        return torch.cat(result,dim=0)
    
    def _aggregate_logits(self, logits, valid_masks):
        """Aggregate logits for each data.

        Args:
            logits (torch.Tensor): [B, T, n_classes]
            valid_masks (torch.Tensor): [B, T]
        """
        if self.agg_func == 'sum':
            logits = logits.sum(1)
        elif self.agg_func == 'mean':
            logits = logits.sum(1) / valid_masks.float().sum(1, keepdim=True)
        elif self.agg_func == 'max':
            # make invalid logits very small
            logits = logits - (1. - valid_masks.float()) * 1e6
            logits = logits.max(1)[0]
        else:
            raise NotImplementedError
        return logits
    
    def _aggregate_probs(self, logits, valid_masks):
        """This one always take the mean."""
        valid_masks = valid_masks.detach().float()
        probs = logits.softmax(dim=-1)
        probs = probs * valid_masks[..., None]
        probs = probs.sum(1) / valid_masks.sum(1, keepdim=True)
        return probs
    
    def forward(self, data_dict):
        imgs=data_dict['img'] # [B, T, C, H, W]
        valid_masks = data_dict['valid_mask'] # [B, T]
        B, T = valid_masks.shape
        
        valid_imgs = imgs[valid_masks] # [N, C, H, W]
        
        img_feats=self.get_event_feats(valid_imgs) # [N, C]
        text_feats=self.get_text_feats()    # [n_classes, C]
        
        n_cls = text_feats.shape[0]
        logits = (img_feats @ text_feats.T)  # [N, n_cls]
        
        full_logits = torch.zeros(B, T, n_cls).type_as(logits)
        full_logits[valid_masks] = logits
        logits = self._aggregate_logits(full_logits, valid_masks)
        probs = self._aggregate_probs(full_logits, valid_masks)
        
        out_dict = {
            'full_logits': full_logits,  # [B, T, n_classes]
            'valid_masks': valid_masks,  # [B, T]
            'logits': logits,  # [B, n_classes]
            'probs': probs,  # [B, n_classes]
        }
        return out_dict
    
    def calc_train_loss(self, data_dict, out_dict):
        labels = data_dict['label'] # [B]
        logits = out_dict['logits']  # [B, n_classes]
        probs = out_dict['probs']  # [B, n_classes]
        loss_dict = {}
        if self.use_logits_loss:
            loss_dict['ce_loss'] = F.cross_entropy(logits, labels)
        if self.use_probs_loss:
            probs = probs + 1e-6  # avoid nan
            loss_dict['ce_loss'] = F.nll_loss(probs.log(), labels)
        
        return loss_dict
    
    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval."""
        loss_dict = self.calc_train_loss(data_dict, out_dict)

        # also compute the cls accuracy
        labels = data_dict['label']  # [B]
        # based on aggregated probs
        probs = out_dict['probs']  # [B, n_classes]
        probs_acc = (probs.argmax(dim=-1) == labels).float().mean()
        loss_dict['probs_acc'] = probs_acc
        # based on aggregated logits
        logits = out_dict['logits']  # [B, n_classes]
        logits_acc = (logits.argmax(dim=-1) == labels).float().mean()
        loss_dict['logits_acc'] = logits_acc
        return loss_dict
    
    
    
        
        
        