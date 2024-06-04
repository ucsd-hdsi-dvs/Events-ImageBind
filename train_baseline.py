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
        for param in self.imagebind.parameters():
            param.requires_grad = False
        
        feature_size = self._get_feature_size()
    
        self.classifier = nn.Linear(feature_size, num_classes)
        
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
         # video shape: (batch=1,seq_len, 3, 2, 224, 224)
        with torch.no_grad():
            video_embd = self.imagebind({"vision":x})['vision'] # batch, 1024
        logits = self.classifier(video_embd) # logit: batch, num_classes
        return logits
    
    def forward_val(self,x):
        # input shape: (batch, seq_len, 3, 2, 224, 224)
        # logits = []
         # video shape: (batch=1,seq_len, 3, 2, 224, 224)
        logits=[]
        batch_size = 1
        num_batches = (x.size(1) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, x.size(1))
            batch = x[:,start_idx:end_idx, :, :, :, :] # batch shape: (1,batch, 3, 2, 224, 224)
            with torch.no_grad():
                video_embd = self.imagebind({"vision":batch})['vision'] # 1, 1024
            logits.append(video_embd) # logits: seq_len, 1024
        # mean logits
        logits=torch.cat(logits, dim=0)
        logits = logits.mean(dim=0, keepdim=True) # logits: 1, 1024 
        logits = self.classifier(logits) # logits: 1, num_classes
        return logits
        
    def training_step(self, batch, batch_idx):
        videos, labels = batch
        logits=self(videos)
        gt=ground_truth_decoder(labels).to(logits.device)
        loss = self.criterion(logits, gt)
        # print(loss)
        self.log('train_loss', loss, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size)
        
        # acc=multi_label_accuracy(logits, gt)
        # self.log('train_acc', acc, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        logits=self.forward_val(videos)
        gt=ground_truth_decoder(labels).to(logits.device)
        loss = self.criterion(logits, gt)
        self.log('val_loss', loss, on_step=False, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size, sync_dist=True)
        
        acc=multi_label_accuracy(logits, gt)
        self.log('val_acc', acc, on_step=False, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size, sync_dist=True)
        
        preds=custom_multi_label_pred(logits)
        self.acc.update(preds, gt)
        self.confusion_matrix.update(preds, gt)
        return loss
    
    def test_step(self, batch, batch_idx):
        videos, labels = batch
        logits=self.forward_val(videos)
        gt=ground_truth_decoder(labels).to(logits.device)
        
        loss = self.criterion(logits, gt)
        self.log('test_loss', loss, on_step=False, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size, sync_dist=True)
        
        acc=multi_label_accuracy(logits, gt)
        self.log('test_acc', acc, on_step=False, on_epoch=LOG_ON_EPOCH,prog_bar=True,batch_size=self.batch_size, sync_dist=True)
        
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
    
    def on_validation_epoch_end(self):
        print('Validation Epoch End')
        
        acc=self.acc.compute()
        self.log('val_acc', acc,batch_size=self.batch_size, on_epoch=True,on_step=False,prog_bar=True, sync_dist=True)
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
        print('Test Epoch End')
        
        acc=self.acc.compute()
        self.log('test_acc', acc, batch_size=self.batch_size, on_epoch=True,on_step=False,prog_bar=True, sync_dist=True)
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

    def _load_checkpoint(self, checkpoint_path):
        checkpoint=torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        
        lstm_state_dict = {k[len('lstm.'):]: v for k, v in state_dict.items() if k.startswith('lstm.')}
        classifier_state_dict = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}
        self.lstm.load_state_dict(lstm_state_dict)
        self.classifier.load_state_dict(classifier_state_dict)
        print("Checkpoint loaded successfully")
    

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
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to load checkpoint')
    
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
    
    if args.load_checkpoint:
        model._load_checkpoint(args.load_checkpoint)
    
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
                      logger=loggers if loggers else None,strategy='ddp', use_distributed_sampler=False,
                       **checkpointing)
    
    trainer.fit(model, dataModule)
    