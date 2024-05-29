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
import matplotlib.pyplot as plt
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module, save_module
from pytorch_loss import FocalLossV3
from eventutils import AccMetric,ConfusionMatrixMetric, multi_label_accuracy, custom_multi_label_pred, ground_truth_decoder


logging.basicConfig(level=logging.INFO, force=True)

# Logging settings
LOG_ON_STEP = True
LOG_ON_EPOCH = True

class VideoTrain(L.LightningModule):
    def __init__(self, num_classes=3,lstm_layers=1,hidden_dim=128,confusion_path='/tsukimi/datasets/Chiba/imagebind_baseline/confusion',prefix='firstrun'):
        super(VideoTrain, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        
        self.imagebind=imagebind_model.imagebind_huge(pretrained=True)
        self.imagebind.eval()
        feature_size = self._get_feature_size()
        
        self.lstm = nn.LSTM(feature_size, hidden_dim, lstm_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.criterion = FocalLossV3()
        self.acc=AccMetric()
        self.confusion_matrix=ConfusionMatrixMetric()
        self.confusion_path=os.path.join(confusion_path, prefix)
    
    def _get_feature_size(self):
        dummy_input = torch.zeros(
                [
                    1,
                ]
                + [3,2, 224, 224]
            )
        with torch.no_grad():
            features = self.imagebind(dummy_input)
        return features.numel()

    def forward(self, x):
        # input shape: (batch, seq_len, 3, 2, 224, 224)
        logits = []
        for video in x: # video shape: (seq_len, 3, 2, 224, 224)
            with torch.no_grad():
                video_embd = self.imagebind(video)  # (seq_len, feature_size)
            lstm_out, _ = self.lstm(video_embd.unsqueeze(0))  # (1, seq_len, hidden_dim) 
            logits.append(self.classifier(lstm_out[:, -1, :]).squeeze(0))  # (num_classes)
        return torch.stack(logits)
        
    def training_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        gt=ground_truth_decoder(labels)
        loss = self.criterion(logits, gt)
        self.log('train_loss_'+'Focal_Loss', loss, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True)
        
        acc=multi_label_accuracy(logits, gt)
        self.log('train_acc', acc, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True)
        
        preds=custom_multi_label_pred(logits)
        self.acc.update(preds, gt)
        return loss
    
    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        gt=ground_truth_decoder(labels)
        loss = self.criterion(logits, gt)
        self.log('val_loss_'+'Focal_Loss', loss, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True)
        
        acc=multi_label_accuracy(logits, gt)
        self.log('val_acc', acc, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True)
        
        preds=custom_multi_label_pred(logits)
        self.acc.update(preds, gt)
        self.confusion_matrix.update(preds, gt)
        return loss
    
    def test_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        gt=ground_truth_decoder(labels)
        loss = self.criterion(logits, gt)
        self.log('test_loss_'+'Focal_Loss', loss, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True)
        
        acc=multi_label_accuracy(logits, gt)
        self.log('test_acc', acc, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True)
        
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
        self.log('train_acc', acc, on_step=False, on_epoch=True,prog_bar=True)
        self.acc.reset()
    
    def on_validation_epoch_end(self):
        acc=self.acc.compute()
        self.log('val_acc', acc, on_step=False, on_epoch=True,prog_bar=True)
        self.acc.reset()
        
        cm=self.confusion_matrix.compute()
        self.log('val_cm', cm, on_step=False, on_epoch=True,prog_bar=True)
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
        self.log('test_acc', acc, on_step=False, on_epoch=True,prog_bar=True)
        self.acc.reset()
        
        cm=self.confusion_matrix.compute()
        self.log('test_cm', cm, on_step=False, on_epoch=True,prog_bar=True)
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
    parser.add_argument("--datasets_dir", type=str, default="./.datasets",
                        help="Directory containing the datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["dreambooth"], choices=["dreambooth"],
                        help="Datasets to use for training and validation")
    parser.add_argument("--full_model_checkpoint_dir", type=str, default="./.checkpoints/full",
                        help="Directory to save the full model checkpoints")
    parser.add_argument("--full_model_checkpointing", action="store_true", help="Save full model checkpoints")
    parser.add_argument("--loggers", type=str, nargs="+", choices=["tensorboard", "wandb", "comet", "mlflow"],
                        help="Loggers to use for logging")
    parser.add_argument("--loggers_dir", type=str, default="./.logs", help="Directory to save the logs")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (Don't plot samples on start)")

    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs to train")
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
    
    