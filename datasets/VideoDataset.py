import os
from typing import Optional, Callable
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import torch
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader


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

class VideoDataset(Dataset):
    def __init__(self,
                 mode='train',
                 frame_step=2,
                 csv_path='/tsukimi/datasets/Chiba/baseline/datalist_3',
                 ):
        self.mode=mode
        self.frame_step=frame_step
        self.csv_path=csv_path
        
        if mode=='train':
            self.data_list = pd.read_csv(os.path.join(csv_path, 'train.csv'),header=None, delimiter=',')
        elif mode=='val':
            self.data_list = pd.read_csv(os.path.join(csv_path, 'val.csv'),header=None, delimiter=',')
        elif mode=='test':
            self.data_list = pd.read_csv(os.path.join(csv_path, 'test.csv'),header=None, delimiter=',')
        
        self.labels=list(self.data_list.values[:,1])
        self.videos=list(self.data_list.values[:,0])
        self.frame_normalize = transforms.Compose([
                resize_pad,
                transforms.Normalize([0.153, 0.153, 0.153], [0.165, 0.165, 0.165])])
        
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        video_path=self.videos[idx]
        label=self.labels[idx]
        
        frames=self._read_video_frames(video_path)
        return frames, label
    
    def _read_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame1 = cap.read()
        count = 0
        
        while success:
            success, frame2 = cap.read()
            if not success:
                break

            if count % self.frame_step == 0:
                # Convert both frames to RGB and tensors, then permute
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                frame1 = torch.tensor(frame1).permute(2, 0, 1).float()/255.0
                frame2 = torch.tensor(frame2).permute(2, 0, 1).float()/255.0
                
                # Normalize the frames
                frame1 = self.frame_normalize(frame1)
                frame2 = self.frame_normalize(frame2)
                
                # Combine frames (e.g., concatenate along the channel dimension)
                combined_frame = torch.stack([frame1, frame2], dim=0) # Shape: (2, 3, H, W)
                
                # transpose to (3,2,H,W)
                combined_frame = combined_frame.permute(1,0,2,3)
                
                frames.append(combined_frame)
            
            frame1 = frame2  # Set the second frame as the first for the next iteration
            count += 1
        cap.release()
        return torch.stack(frames) # L,C,T,H,W
        

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, batch_size=2, frame_step=2):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.frame_step = frame_step
    
    def train_dataloader(self) -> os.Any:
        dataset=VideoDataset(mode='train', frame_step=self.frame_step, csv_path=self.csv_path)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
    
    def val_dataloader(self) -> os.Any:
        dataset=VideoDataset(mode='val', frame_step=self.frame_step, csv_path=self.csv_path)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
    
    def test_dataloader(self) -> os.Any:
        dataset=VideoDataset(mode='test', frame_step=self.frame_step, csv_path=self.csv_path)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
    
    def collate_fn(self, batch):
        videos = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # B, C, T, H, W
        return videos, torch.tensor(labels)
        
        