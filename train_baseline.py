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
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module, save_module
from pytorch_loss import FocalLossV3
from eventutils import AccMetric,ConfusionMatrixMetric, multi_label_accuracy, custom_multi_label_pred, ground_truth_decoder
from datasets.VideoDataset import VideoDataModule
import seaborn as sns

logging.basicConfig(level=logging.INFO, force=True)

# Logging settings
LOG_ON_STEP = True
LOG_ON_EPOCH = True

class VideoTrain(L.LightningModule):
    def __init__(self, batch_size=1, num_classes=3,lstm_layers=1,hidden_dim=128,confusion_path='/tsukimi/datasets/Chiba/imagebind_baseline/confusion',prefix='firstrun',
                 lr=5e-4, weight_decay=1e-4, max_epochs=500,momentum_betas=(0.9, 0.95)):
        super(VideoTrain, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        self.imagebind=imagebind_model.imagebind_huge(pretrained=True)
        self.imagebind.eval()
        feature_size = self._get_feature_size()
        
        self.lstm = nn.LSTM(feature_size, hidden_dim, lstm_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.criterion = FocalLossV3()
        self.acc=AccMetric()
        self.confusion_matrix=ConfusionMatrixMetric()
        
        self.confusion_path=os.path.join(confusion_path, prefix)
        if not os.path.exists(self.confusion_path):
            os.makedirs(self.confusion_path, exist_ok=True)
    
    def _get_feature_size(self):
        dummy_input = torch.zeros(
                [
                    1,1
                ]
                + [3,2, 224, 224]
            )
        with torch.no_grad():
            features = self.imagebind({"vision":dummy_input})
        return features['vision'].shape[-1]

    def forward(self, x):
        # input shape: (batch, seq_len, 3, 2, 224, 224)
        # logits = []
         # video shape: (batch,seq_len, 3, 2, 224, 224)
        with torch.no_grad():
            video_embd = self.imagebind({"vision":x})['vision']  # (batch,seq_len, feature_size)
        lstm_out, _ = self.lstm(video_embd)  # (batch, seq_len, hidden_dim) 
        # logits.append(self.classifier(lstm_out[:, -1, :]))  # (batch,num_classes)
        return self.classifier(lstm_out[:, -1, :])
        
    def training_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        gt=ground_truth_decoder(labels).to(logits.device)
        loss = self.criterion(logits, gt)
        self.log('train_loss_'+'Focal_Loss', loss, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size)
        
        acc=multi_label_accuracy(logits, gt)
        self.log('train_acc', acc, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size)
        
        preds=custom_multi_label_pred(logits)
        self.acc.update(preds, gt)

        return loss
    
    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        gt=ground_truth_decoder(labels).to(logits.device)
        loss = self.criterion(logits, gt)
        # self.log('val_loss', loss, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size)
        
        acc=multi_label_accuracy(logits, gt)
        # self.log('val_acc', acc, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size)
        
        preds=custom_multi_label_pred(logits)
        self.acc.update(preds, gt)
        self.confusion_matrix.update(preds, gt)
        return loss
    
    def test_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        gt=ground_truth_decoder(labels).to(logits.device)
        
        loss = self.criterion(logits, gt)
        # self.log('test_loss_'+'Focal_Loss', loss, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size)
        
        acc=multi_label_accuracy(logits, gt)
        # self.log('test_acc', acc, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size)
        
        preds=custom_multi_label_pred(logits)
        self.acc.update(preds, gt)
        self.confusion_matrix.update(preds, gt)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, 
                                betas=self.hparams.momentum_betas)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]
    
    def on_train_epoch_end(self):
        acc=self.acc.compute()
        self.log('train_acc', acc,batch_size=self.batch_size)
        self.acc.reset()
    
    def on_validation_epoch_end(self):
        print('Validation Epoch End')
        acc=self.acc.compute()
        self.log('val_acc', acc,batch_size=self.batch_size)
        self.acc.reset()
        
        cm=self.confusion_matrix.compute()
        # self.log('val_cm', cm, on_step=False, on_epoch=True,prog_bar=True)
        self.confusion_matrix.reset()
        
        save_path=os.path.join(self.confusion_path, 'val_'+str(self.current_epoch)+'.png')
        plt.figure(figsize=(10,10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()
        
    
    def on_test_epoch_end(self):
        acc=self.acc.compute()
        self.log('test_acc', acc, batch_size=self.batch_size)
        self.acc.reset()
        
        cm=self.confusion_matrix.compute()
        # self.log('test_cm', cm, batch_size=self.batch_size)
        self.confusion_matrix.reset()
        
        save_path=os.path.join(self.confusion_path, 'test.png')
        plt.figure(figsize=(10,10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ImageBind model with PyTorch Lightning and LoRA.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training ('cpu' or 'cuda')")
    parser.add_argument("--full_model_checkpoint_dir", type=str, default="./.checkpoints/full",
                        help="Directory to save the full model checkpoints")
    parser.add_argument("--full_model_checkpointing", action="store_true", help="Save full model checkpoints")
    parser.add_argument("--loggers", type=str, nargs="+", choices=["tensorboard", "wandb", "comet", "mlflow"],
                        help="Loggers to use for logging")
    parser.add_argument("--loggers_dir", type=str, default="./.logs", help="Directory to save the logs")

    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--momentum_betas", nargs=2, type=float, default=[0.9, 0.95],
                        help="Momentum beta 1 and 2 for Adam optimizer")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for InfoNCE loss")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--self_contrast", action="store_true", help="Use self-contrast on the image modality")
    
    parser.add_argument("--lstm_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of the LSTM")
    parser.add_argument("--confusion_path", type=str, default='/tsukimi/datasets/Chiba/imagebind_baseline/confusion', help="Path to save confusion matrix")
    parser.add_argument("--prefix", type=str, default='firstrun', help="Prefix for confusion matrix")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create loggers
    loggers = []
    for logger in args.loggers if args.loggers is not None else []:
        if logger == "wandb":
            wandb.init(project="imagebind", config=args)
            wandb_logger = pl_loggers.WandbLogger(
                save_dir=args.loggers_dir,
                name="imagebind")
            loggers.append(wandb_logger)
        elif logger == "tensorboard":
            tensorboard_logger = pl_loggers.TensorBoardLogger(
                save_dir=args.loggers_dir,
                name="imagebind")
            loggers.append(tensorboard_logger)
        elif logger == "comet":
            comet_logger = pl_loggers.CometLogger(
                save_dir=args.loggers_dir,
                api_key=os.environ["COMET_API_KEY"],
                workspace=os.environ["COMET_WORKSPACE"],
                project_name=os.environ["COMET_PROJECT_NAME"],
                experiment_name=os.environ.get("COMET_EXPERIMENT_NAME", None),
            )
            loggers.append(comet_logger)
        elif logger == "mlflow":
            mlflow_logger = pl_loggers.MLFlowLogger(
                save_dir=args.loggers_dir,
                experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"],
                tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
                run_name="imagebind"
            )
            loggers.append(mlflow_logger)
        else:
            raise ValueError(f"Unknown logger: {logger}")
    
    # Set experiment properties
    seed_everything(args.seed, workers=True)
    torch.backends.cudnn.determinstic = True
    device_name = args.device  # "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    
    dataModule=VideoDataModule(csv_path='/tsukimi/datasets/Chiba/baseline/datalist_3',batch_size=args.batch_size)
    
    model=VideoTrain(num_classes=3,lstm_layers=args.lstm_layers,hidden_dim=args.hidden_dim,confusion_path=args.confusion_path,prefix=args.prefix,
                     max_epochs=args.max_epochs,lr=args.lr, weight_decay=args.weight_decay, momentum_betas=args.momentum_betas,batch_size=args.batch_size)
    if args.full_model_checkpointing:
        checkpointing = {"enable_checkpointing": args.full_model_checkpointing,
                         "callbacks": [ModelCheckpoint(monitor="val_acc", dirpath=args.full_model_checkpoint_dir,
                                                        filename="imagebind-{epoch:02d}-{val_acc:.2f}",
                                                        save_last=True, mode="max")]}
    else:
        checkpointing = {"enable_checkpointing": args.full_model_checkpointing,}
    
    trainer=Trainer(accelerator="gpu" if "cuda" in device_name else "cpu",
                    devices=1 if ":" not in device_name else int(device_name.split(":")[1]), deterministic=True,
                      max_epochs=args.max_epochs, gradient_clip_val=args.gradient_clip_val,
                      logger=loggers if loggers else None,
                       **checkpointing)
    
    trainer.fit(model, dataModule)
    