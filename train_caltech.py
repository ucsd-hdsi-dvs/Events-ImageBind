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
from datasets.eclipdatasets.event2img import Event2ImageDataset
from models.imagebind_model import imagebind_huge
from models.events import EventModel
import copy
from models.adapter import IdentityAdapter, TransformerAdapter

logging.basicConfig(level=logging.INFO, force=True)

# Logging settings
LOG_ON_STEP = True
LOG_ON_EPOCH = True


def resize_pad(frame, size=224):
    """
    resize a frame's longer side to 224, pad the shorter side to 224
    """

    # get shape
    c, h, w = frame.shape

    # get longer side
    longer_side = max(h, w)

    # calculate ratio
    ratio = size / longer_side

    # resize with transform
    resize_transform = transforms.Resize((int(h * ratio), int(w * ratio)))
    frame = resize_transform(frame)

    # get new shape
    c, h, w = frame.shape

    # calculate padding needed to reach size for both dimensions
    pad_height = (size - h) if h < size else 0
    pad_width = (size - w) if w < size else 0

    # calculate padding for each side to center the image
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # apply padding
    padding_transform = transforms.Pad(padding=(pad_left, pad_top, pad_right, pad_bottom), fill=0, padding_mode='constant')
    frame = padding_transform(frame)

    return frame

def load_model_from_checkpoint(checkpoint_path='/tsukimi/datasets/Chiba/finetune_checkpoint/checkpoints/imgbind_fintuned.pth',device=None):
    """
    Load a model from a checkpoint.
    """
    model=imagebind_huge(pretrained=True)
    eventmodel=EventModel()
    eventmodel.apply_event_layers(model)
    # load the state dict
    model.load_state_dict(torch.load(checkpoint_path))
    return model
    

class CaltechTrain(L.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=1e-4, max_epochs=500, batch_size=32, num_workers=4, seed=42, 
                 self_contrast=False, temperature=0.07,  momentum_betas=(0.9, 0.95), checkpoint_path=None,class_names=None, prompt='a point cloud image of a {}',
                 adapter_dict=dict(
                # 'trans', 'identity'
                # 'text-{}' with the above: tune text features as FC weight
                adapter_type='trans',
                residual=True,
                in_dim=1024,
            )
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model=load_model_from_checkpoint(checkpoint_path)
        self.model.eval()
        
        # for modality_preprocessor in self.model.modality_preprocessors.children():
        #     modality_preprocessor.requires_grad_(False)
        # for modality_trunk in self.model.modality_trunks.children():
        #     modality_trunk.requires_grad_(False)
        
        self.class_names=class_names
        self.prompt=prompt
        self.text_feats=None
        self.agg_func='mean'
        self.adapter_dict = copy.deepcopy(adapter_dict)
        
        self._build_adapter()
    
    def _build_prompts(self, adapter_type):
        """Build the text features for prompt tuning."""
        with torch.no_grad():
            text_feats = self.get_text_feats().float()  # [n_classes, C]
        self.text_feats =torch.nn.Parameter(text_feats, requires_grad=True)
        adapter_type = adapter_type[5:]
        return adapter_type
    
    def _build_adapter(self):
        # whether to tune the text features as well
        adapter_type = self.adapter_dict.pop('adapter_type').lower()
        if adapter_type.startswith('text-'):
            print('Tune text features as well!')
            self.prompt_tuning = True
            adapter_type = self._build_prompts(adapter_type)
        else:
            self.prompt_tuning = False

        # image feature adapter
        self.adapter_type = adapter_type
        if adapter_type == 'identity':  # not tuning image features
            model = IdentityAdapter
        elif adapter_type == 'trans':  # Transformer to fuse image features
            model = TransformerAdapter
        else:
            raise NotImplementedError(f'adapter {adapter_type} not supported!')
        self.adapter = model(**self.adapter_dict)
    
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
        if self.text_feats is not None:
            return self.text_feats
        
        class_names = [c.lower().replace('_', ' ') for c in self.class_names]
        prompts=torch.cat([tokenize(self.prompt.format(c)) for c in class_names]).to(self.device)
        
        result=[]
        for text in prompts:
            with torch.no_grad():
                text_feats = self.model({ModalityType.TEXT: text.unsqueeze(0)})
            text_feats=list(text_feats.values())[0]
            result.append(text_feats)
        result=torch.cat(result,dim=0)
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
        
        # update image features using adapter
        C = img_feats.shape[-1]
        full_img_feats = torch.zeros(B, T, C).type_as(img_feats)
        full_img_feats[valid_masks] = img_feats
        full_img_feats = self.adapter(full_img_feats, valid_masks)
        full_img_feats = F.normalize(
            full_img_feats, p=2, dim=-1).type_as(full_img_feats)
        full_img_feats = full_img_feats * valid_masks.float().unsqueeze(-1)
        
        n_cls = text_feats.shape[0]
        full_logits = (full_img_feats @ text_feats.T)  # [N, n_cls]
        
        # print("full_img_feats",full_img_feats.shape)
        # print("full_logits",full_logits.shape)
        
        # full_logits = torch.zeros(B, T, n_cls).type_as(logits)
        # full_logits[valid_masks] = logits
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
        probs_acc = (probs.argmax(dim=-1) == labels).float().mean()
        
        loss_dict = {}
        loss_dict['ce_loss'] = F.cross_entropy(logits, labels)
        loss_dict['probs_acc'] = probs_acc
        
        self.log("train_loss", loss_dict['ce_loss'], on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,sync_dist=True)
        self.log("train_acc", probs_acc, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,sync_dist=True)
        return loss_dict
    
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
        
        self.log("val_loss", loss_dict['ce_loss'], on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,sync_dist=True)
        self.log("val_acc", probs_acc, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,sync_dist=True)
        return loss_dict
    
    def training_step(self, batch, batch_idx):
        out_dict = self(batch)
        loss_dict = self.calc_train_loss(batch, out_dict)
        return loss_dict['ce_loss']
    
    def validation_step(self, batch, batch_idx):
        out_dict = self(batch)
        loss_dict = self.calc_eval_loss(batch, out_dict)
        return loss_dict['ce_loss']


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ImageBind model classifier with PyTorch Lightning")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training ('cpu' or 'cuda')")
    parser.add_argument("--datasets_dir", type=str, default="./.datasets",
                        help="Directory containing the datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default="caltech", choices=["caltech"],
                        help="Datasets to use for training and validation")
    parser.add_argument("--loggers", type=str, nargs="+", choices=["tensorboard", "wandb", "comet", "mlflow"],
                        help="Loggers to use for logging")
    parser.add_argument("--loggers_dir", type=str, default="./.logs", help="Directory to save the logs")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--momentum_betas", nargs=2, type=float, default=[0.9, 0.95],
                        help="Momentum beta 1 and 2 for Adam optimizer")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for InfoNCE loss")
    parser.add_argument('--N', type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint to load and continue training")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    wandb_logger = pl_loggers.WandbLogger(
                project="imagebind_finetune")
    seed_everything(args.seed, workers=True)
    torch.backends.cudnn.determinstic = True
    device_name = args.device  # "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    
    data_transform=frame_normalize = transforms.Compose([
                transforms.ToTensor(),
                resize_pad])
    quantize_args=dict(
            max_imgs=2,
            split_method='event_count',
            convert_method='event_histogram',
            N=20000,
            grayscale=False,
            count_non_zero=False,  # hotpixel statistics
            background_mask=True,  # apply white background via alpha-masking
        )
    
    if args.datasets=="caltech":
        from datasets.eclipdatasets.caltech import NCaltech101,NEW_CNAMES
        
        train_dataset=NCaltech101(os.path.join(args.datasets_dir,'training'), augmentation=True, new_cnames=NEW_CNAMES)
        class_names=train_dataset.classes
        train_dataset=Event2ImageDataset(transforms=data_transform, event_dataset=train_dataset,quantize_args=quantize_args,augment=False,tta=False)
        
        val_dataset=NCaltech101(os.path.join(args.datasets_dir,'testing'), augmentation=False, new_cnames=NEW_CNAMES)
        val_dataset=Event2ImageDataset(transforms=data_transform, event_dataset=val_dataset,quantize_args=quantize_args,augment=False,tta=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model=CaltechTrain(max_epochs=args.max_epochs, batch_size=args.batch_size, lr=args.lr,
                           weight_decay=args.weight_decay, momentum_betas=args.momentum_betas,
                           temperature=args.temperature,class_names=class_names,checkpoint_path='/root/Events-ImageBind/.checkpoints/imgbind_fintuned.pth')
    checkpointing = {"enable_checkpointing": True,
                         "callbacks": [ModelCheckpoint(monitor="val_acc", dirpath="/tsukimi/datasets/Chiba/finetune_checkpoint/caltech",
                                                        filename="imagebind-{epoch:02d}-{val_loss:.2f}",
                                                        save_last=True, mode="min")]}
    
    trainer=Trainer(accelerator="gpu" if "cuda" in device_name else "cpu",
                      devices=1 if ":" not in device_name else int(device_name.split(":")[1]), deterministic=True,
                      max_epochs=args.max_epochs,gradient_clip_val=args.gradient_clip_val,
                      logger=wandb_logger, **checkpointing,strategy="ddp_find_unused_parameters_true")
    trainer.fit(model, train_loader, val_loader)
    
    

# python train_caltech.py 


    
    
        
        
        