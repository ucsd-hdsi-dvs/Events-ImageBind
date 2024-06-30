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
                 self_contrast=False, temperature=0.07,  momentum_betas=(0.9, 0.95), checkpoint_path=None
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model=load_model_from_checkpoint(checkpoint_path)
        self.model.eval()
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, 
                                betas=self.hparams.momentum_betas)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]
    
    def get_text_feats(self, text):
        with torch.no_grad():
            text_feats = self.model({ModalityType.TEXT: tokenize(text)})
        text_feats=list(text_feats.values())[0]
        return text_feats
    
    def get_event_feats(self, events):
        with torch.no_grad():
            event_feats = self.model({ModalityType.EVENT: events.unsqueeze(0)})
        event_feats=list(event_feats.values())[0]
        return event_feats
    
    def calculate_loss(self, batch):
        event_frames=batch['img']
        texts=batch['prompt']
        
        
        