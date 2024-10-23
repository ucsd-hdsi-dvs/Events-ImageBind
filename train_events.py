# Based on PyTorch Lightning Tutorial 13 -
# SSL : https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
# Modified by Fares Abawi (@fabawi).
import logging
import os
import argparse
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
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
from pytorch_lightning.strategies import DDPStrategy

from models import imagebind_model
from models import lora as LoRA
from models.imagebind_model import ModalityType, load_module, save_module
from models.events import EventModel

logging.basicConfig(level=logging.INFO, force=True)

# Logging settings
LOG_ON_STEP = True
LOG_ON_EPOCH = True


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class ImageBindTrain(L.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=1e-4, max_epochs=500, batch_size=32, num_workers=4, seed=42, 
                 self_contrast=False, temperature=0.07,  momentum_betas=(0.9, 0.95), 
                 lora=False, lora_rank=4, lora_checkpoint_dir="./.checkpoints/lora",
                 lora_layer_idxs=None, lora_modality_names=None,
                 linear_probing=False, checkpoint_path=None, load_vision_to_event=True,
                 ):
        super().__init__()
        assert not (linear_probing and lora), \
            "Linear probing is a subset of LoRA training procedure for ImageBind. " \
            "Cannot set both linear_probing=True and lora=True. " \
            "Linear probing stores params in lora_checkpoint_dir"
        self.save_hyperparameters()

        # Load full pretrained ImageBind model
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        
        # apply event layer
        eventmodel=EventModel()
        eventmodel.apply_event_layers(self.model, load_vision_to_event)
        
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            modality_state_dict = {
                k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()
            }
            model_state_dict = self.model.state_dict()
            model_state_dict.update(modality_state_dict)
            self.model.load_state_dict(model_state_dict)
            
        
        for modality_preprocessor in self.model.modality_preprocessors.children():
            modality_preprocessor.requires_grad_(False)


        # for modality_trunk in self.model.modality_trunks.children():
        #     modality_trunk.requires_grad_(False)
        # freeze vision channels
        for params in self.model.modality_trunks[ModalityType.VISION].parameters():
            params.requires_grad_(False)
        for params in self.model.modality_postprocessors[ModalityType.VISION].parameters():
            params.requires_grad_(False)
        for params in self.model.modality_heads[ModalityType.VISION].parameters():
            params.requires_grad_(False)
        
        for params in self.model.modality_preprocessors[ModalityType.EVENT].parameters():
            if not load_vision_to_event:
                params.requires_grad_(True)
                print('unfreezing event preprocessor')
        
        if lora:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)
            
            # add LoRA trunks to the model
            self.model.modality_trunks.update(LoRA.apply_lora_modality_trunks(self.model.modality_trunks, rank=lora_rank,
                                                                              layer_idxs=lora_layer_idxs,
                                                                              modality_names=lora_modality_names))
            
            # Load LoRA checkpoint
            LoRA.load_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=lora_checkpoint_dir)

            # Load postprocessors & heads
            load_module(self.model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=lora_checkpoint_dir)
            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=lora_checkpoint_dir)
        # require grad for last layer of each modality head
        elif linear_probing:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)
            for modality_postprocessor in self.model.modality_postprocessors.children():
                modality_postprocessor.requires_grad_(False)

            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=lora_checkpoint_dir)
            for modality_head in self.model.modality_heads.children():
                modality_head.requires_grad_(False)
                final_layer = list(modality_head.children())[-1]
                final_layer.requires_grad_(True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, 
                                betas=self.hparams.momentum_betas)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        data_a, class_a, data_b, class_b = batch

        # class_a is always "vision" according to ImageBind
        # feats_a = [self.model({class_a[0]: data_a_i.unsqueeze(0)}) for data_a_i in data_a]
        # feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)
        with torch.no_grad():
            feats_a_tensor=list(self.model({class_a[0]: data_a}).values())[0]

        # class_b could be any modality
        # feats_b = [self.model({class_b[idx]: data_b_i.unsqueeze(0)}) for idx, data_b_i in enumerate(data_b)]
        # feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)

        feats_b_tensor=list(self.model({class_b[0]: data_b}).values())[0]
        
        if self.hparams.self_contrast:
            feats_a_b_tensor = torch.cat([feats_a_tensor.chunk(2)[0], feats_b_tensor], dim=0)
            feats_tensors = [feats_a_tensor, feats_a_b_tensor]
            temperatures = [1, self.hparams.temperature]
            contrast = ["self", "cross"]
        else:
            feats_a_b_tensor = torch.cat([feats_a_tensor, feats_b_tensor], dim=0)
            feats_tensors = [feats_a_b_tensor]
            temperatures = [self.hparams.temperature]
            contrast = ["cross"]
        
        # Accumulate self-contrastive loss for image and its augmentation, and modailty with image
        dual_nll = False
        for feats_idx, feats_tensor in enumerate(feats_tensors):
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(feats_tensor[:, None, :], feats_tensor[None, :, :], dim=-1)
            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -9e15)
            # Find positive example -> batch_size//2 away from the original example
            pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
            # InfoNCE loss
            cos_sim = cos_sim / temperatures[feats_idx]
            nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            nll = nll.mean()
            if not dual_nll:
                dual_nll = nll
            else:
                dual_nll += nll
                dual_nll /= 2
            # Logging loss
            self.log(mode + "_loss_" + contrast[feats_idx], nll, prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size, sync_dist=True)
            # Get ranking position of positive example
            comb_sim = torch.cat(
                [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
                dim=-1,
            )
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            # Logging ranking metrics
            self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size, sync_dist=True)
            self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size, sync_dist=True)
            self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size, sync_dist=True)

        self.log(mode + "_loss", dual_nll, prog_bar=True,
                 on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size, sync_dist=True)
        return dual_nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

    def on_validation_epoch_end(self):
        if self.hparams.lora:
            # Save LoRA checkpoint
            LoRA.save_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=self.hparams.lora_checkpoint_dir)
            # Save postprocessors & heads
            save_module(self.model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
            save_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
        elif self.hparams.linear_probing:
            # Save postprocessors & heads
            save_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ImageBind model with PyTorch Lightning and LoRA.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training ('cpu' or 'cuda')")
    # parser.add_argument("--datasets_dir", type=str, default="./.datasets",
    #                     help="Directory containing the datasets")
    parser.add_argument("--load_vision_to_event", action="store_true", help="Load vision to event layers")
    parser.add_argument("--datasets", type=str, nargs="+", default=["rgb_like"], choices=["dreambooth","event","mvsce"],
                        help="Datasets to use for training and validation")
    parser.add_argument("--full_model_checkpoint_dir", type=str, default="./.checkpoints/full",
                        help="Directory to save the full model checkpoints")
    parser.add_argument("--full_model_checkpointing", action="store_true", help="Save full model checkpoints")
    parser.add_argument("--loggers", type=str, nargs="+", choices=["tensorboard", "wandb", "comet", "mlflow"],
                        help="Loggers to use for logging")
    parser.add_argument("--loggers_dir", type=str, default="./.logs", help="Directory to save the logs")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (Don't plot samples on start)")

    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--momentum_betas", nargs=2, type=float, default=[0.9, 0.95],
                        help="Momentum beta 1 and 2 for Adam optimizer")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for InfoNCE loss")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--self_contrast", action="store_true", help="Use self-contrast on the image modality")

    parser.add_argument("--lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LoRA layers")
    parser.add_argument("--lora_checkpoint_dir", type=str, default="./.checkpoints/lora",
                        help="Directory to save LoRA checkpoint")
    parser.add_argument("--lora_modality_names", nargs="+", type=str, default=["vision", "event"],
                        choices=["vision", "text", "audio", "thermal", "depth", "imu","event"],
                        help="Modality names to apply LoRA")
    parser.add_argument("--lora_layer_idxs", nargs="+", type=int,
                        help="Layer indices to apply LoRA")
    parser.add_argument("--lora_layer_idxs_vision", nargs="+", type=int,
                        help="Layer indices to apply LoRA for vision modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_text", nargs="+", type=int,
                        help="Layer indices to apply LoRA for text modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_audio", nargs="+", type=int,
                        help="Layer indices to apply LoRA for audio modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_thermal", nargs="+", type=int,
                        help="Layer indices to apply LoRA for thermal modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_depth", nargs="+", type=int,
                        help="Layer indices to apply LoRA for depth modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_imu", nargs="+", type=int,
                        help="Layer indices to apply LoRA for imu modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_event", nargs="+", type=int,
                        help="Layer indices to apply LoRA for event modality. Overrides lora_layer_idxs if specified")

    parser.add_argument("--linear_probing", action="store_true",
                        help="Freeze model and train the last layers of the head for each modality.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint to load and continue training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    wandb_logger = pl_loggers.WandbLogger(
                project="imagebind_finetune")
    
    
    # Set experiment properties
    seed_everything(args.seed, workers=True)
    torch.backends.cudnn.determinstic = True
    device_name = args.device  # "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    contrast_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=224),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    train_datasets = []
    test_datasets = []

    # Load datasets
    if "dreambooth" in args.datasets:
        from datasets.dreambooth import DreamBoothDataset
        train_datasets.append(DreamBoothDataset(
            root_dir=os.path.join(args.datasets_dir, "dreambooth", "dataset"), split="train",
            transform=ContrastiveTransformations(contrast_transforms,
                                                 n_views=2 if args.self_contrast else 1)))
        test_datasets.append(DreamBoothDataset(
            root_dir=os.path.join(args.datasets_dir, "dreambooth", "dataset"), split="test",
            transform=ContrastiveTransformations(contrast_transforms,
                                                 n_views=2 if args.self_contrast else 1)))
    
    if "event" in args.datasets:
        from datasets.EventDataset import EventDataset
        train_datasets.append(EventDataset(data_dir=args.datasets_dir, mode="train"))
        test_datasets.append(EventDataset(data_dir=args.datasets_dir, mode="test"))
    
    if "mvsce" in args.datasets:
        from datasets.MVSCEDataset import MVSCEDataset
        train_datasets.append(MVSCEDataset(data_dir=args.datasets_dir, mode="train"))
        test_datasets.append(MVSCEDataset(data_dir=args.datasets_dir, mode="test"))
    
    # add rgb-like dataset
    if "rgb_like" in args.datasets:
        from datasets.RGBLikeDataset import RGBLikeDataset
        train_datasets.append(RGBLikeDataset(data_root='/eastdata/datasets/MVSEC/', mode='train'))
        test_datasets.append(RGBLikeDataset(data_root='/eastdata/datasets/MVSEC/', mode= 'test'))
        
    # add event dataset
    if len(args.datasets) == 1:
        train_dataset = train_datasets[0]
        test_dataset = test_datasets[0]
    else:
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Parse indices of layers to apply LoRA
    lora_layer_idxs = {}
    lora_modality_names = []
    modalities = ["vision", "text", "audio", "thermal", "depth", "imu","event"]
    for modality_name in args.lora_modality_names:
        if modality_name in modalities:
            modality_type = getattr(ModalityType, modality_name.upper())
            lora_layer_idxs[modality_type] = getattr(args, f'lora_layer_idxs_{modality_name}', None)
            if not lora_layer_idxs[modality_type]:
                lora_layer_idxs[modality_type] = None
            lora_modality_names.append(modality_type)
        else:
            raise ValueError(f"Unknown modality name: {modality_name}")

    # Train dataset
    model = ImageBindTrain(max_epochs=args.max_epochs, batch_size=args.batch_size, lr=args.lr,
                           weight_decay=args.weight_decay, momentum_betas=args.momentum_betas,
                           temperature=args.temperature,
                           num_workers=args.num_workers, self_contrast=args.self_contrast,
                           lora=args.lora, lora_rank=args.lora_rank, lora_checkpoint_dir=args.lora_checkpoint_dir,
                           lora_layer_idxs=lora_layer_idxs if lora_layer_idxs else None,
                           lora_modality_names=lora_modality_names if lora_modality_names else None,
                           linear_probing=args.linear_probing,
                           load_vision_to_event=args.load_vision_to_event,)
     

    if args.full_model_checkpointing:
        checkpointing = {"enable_checkpointing": args.full_model_checkpointing,
                         "callbacks": [ModelCheckpoint(monitor="val_loss", dirpath=args.full_model_checkpoint_dir,
                                                        filename="imagebind-{epoch:02d}-{val_loss:.2f}",
                                                        save_last=True, mode="min")]}
    else:
        checkpointing = {"enable_checkpointing": args.full_model_checkpointing,}

    # trainer = Trainer(accelerator="gpu" if "cuda" in device_name else "cpu",
    #                   devices=1 if ":" not in device_name else int(device_name.split(":")[1]), deterministic=True,
    #                   max_epochs=args.max_epochs, gradient_clip_val=args.gradient_clip_val,
    #                   logger=wandb_logger, strategy='ddp_find_unused_parameters_true', **checkpointing)

    trainer = Trainer(accelerator="gpu" if "cuda" in device_name else "cpu",
                      devices=4, deterministic=True, precision=16,
                      max_epochs=args.max_epochs, gradient_clip_val=args.gradient_clip_val,
                      logger=wandb_logger, strategy="ddp", **checkpointing)
 
    if args.checkpoint_path is None:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint_path)

