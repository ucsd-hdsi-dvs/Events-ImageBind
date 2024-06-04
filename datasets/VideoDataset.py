import os
from typing import Optional, Callable
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import torch
from torchvision import transforms
import lightning as pl
from torch.utils.data import DataLoader,DistributedSampler,Sampler
from catalyst.data.sampler import DistributedSamplerWrapper
import pickle

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
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame1 = torch.tensor(frame1).permute(2, 0, 1).float()/255.0
        frame1 = self.frame_normalize(frame1)
        
        count = 0
        try:
            while success:
                success, frame2 = cap.read()
                if not success:
                    break

                # Convert both frames to RGB and tensors, then permute
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                frame2 = torch.tensor(frame2).permute(2, 0, 1).float()/255.0
                # Normalize the frames
                frame2 = self.frame_normalize(frame2)
                
                # Combine frames (e.g., concatenate along the channel dimension)
                combined_frame = torch.stack([frame1, frame2], dim=0) # Shape: (2, 3, H, W)
                
                # transpose to (3,2,H,W)
                combined_frame = combined_frame.permute(1,0,2,3)
                
                frames.append(combined_frame)
                
                frame1 = frame2  # Set the second frame as the first for the next iteration
                count += 1
        except Exception as e:
            print(f"Error processing video frames: {e}")
        finally:
            cap.release()
        
        if not frames:
            print(f"No frames found for video at path: {video_path}")
            
        return torch.stack(frames) # L,C,T,H,W

class TrainVideoDataset(VideoDataset):
    def __init__(self,
                 mode='train',
                 frame_step=2,
                 csv_path='/tsukimi/datasets/Chiba/baseline/datalist_3',
                 ):
        self.mode=mode
        self.frame_step=frame_step
        self.csv_path=csv_path
        
        self.data_list = pd.read_csv(os.path.join(csv_path, 'train_16.csv'),header=None, delimiter=',')
        self.labels=list(self.data_list.values[:,1])
        self.videos=list(self.data_list.values[:,0])
        self.start_frame=list(self.data_list.values[:,2])
        self.end_frame=list(self.data_list.values[:,3])
        self.frame_normalize = transforms.Compose([
                resize_pad,
                transforms.Normalize([0.153, 0.153, 0.153], [0.165, 0.165, 0.165])])
    
    def __getitem__(self, idx):
        video_path=self.videos[idx]
        label=self.labels[idx]
        start_frame=self.start_frame[idx]
        end_frame=self.end_frame[idx]
        
        frames=self._read_video_frames(video_path, start_frame, end_frame)
        return frames, label
    
    def _read_video_frames(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        # read from start_frame to end_frame
        start_frame=float(start_frame)
        end_frame=float(end_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        
        success, frame1 = cap.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame1 = torch.tensor(frame1).permute(2, 0, 1).float()/255.0
        frame1 = self.frame_normalize(frame1)
        
        count = 0
        try:
            while success:
                success, frame2 = cap.read()
                if not success:
                    break

                # Convert both frames to RGB and tensors, then permute
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                frame2 = torch.tensor(frame2).permute(2, 0, 1).float()/255.0
                # Normalize the frames
                frame2 = self.frame_normalize(frame2)
                
                # Combine frames (e.g., concatenate along the channel dimension)
                combined_frame = torch.stack([frame1, frame2], dim=0) # Shape: (2, 3, H, W)
                
                # transpose to (3,2,H,W)
                combined_frame = combined_frame.permute(1,0,2,3)
                
                frames.append(combined_frame)
                
                frame1 = frame2  # Set the second frame as the first for the next iteration
                count += 1
                if count == (end_frame-start_frame):
                    break
        except Exception as e:
                print(f"Error processing video frames: {e}")
        finally:
                cap.release()
        
        if not frames:
            print(f"No frames found for video at path: {video_path}")
        
        assert len(frames)==(end_frame-start_frame)
        
        return torch.stack(frames) # L,C,T,H,W


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, csv_path='/tsukimi/datasets/Chiba/baseline/datalist_3', batch_size=1, frame_step=2,val_batch_size=1):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.frame_step = frame_step
        self.data_distribution={'others': 117037, 'restrainer_interaction': 82163, 'interaction_with_partner': 34916}
    
    def prepare_data(self):
        pass
    
    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = TrainVideoDataset(mode='train', frame_step=self.frame_step, csv_path=self.csv_path)
            self.val_dataset = VideoDataset(mode='val', frame_step=self.frame_step, csv_path=self.csv_path)
            self.weights=self.load_weights()
        elif stage == 'test':
            self.test_dataset = VideoDataset(mode='test', frame_step=self.frame_step, csv_path=self.csv_path)
    
    def train_dataloader(self):
        num_tasks=self.trainer.world_size
        global_rank = self.trainer.global_rank
        weighted_sampler=torch.utils.data.WeightedRandomSampler(self.weights,len(self.weights))
        self.sampler_train=DistributedSamplerWrapper(sampler=weighted_sampler,num_replicas=num_tasks,rank=global_rank)
        
        return DataLoader(self.train_dataset, batch_size=self.batch_size,collate_fn=self.collate_fn, num_workers=4,sampler=self.sampler_train)
    
    def val_dataloader(self):
        num_tasks=self.trainer.world_size
        global_rank = self.trainer.global_rank
        distributed_sampler = DistributedSampler(self.val_dataset, num_replicas=num_tasks, rank=global_rank)
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, collate_fn=self.collate_fn, num_workers=4,sampler=distributed_sampler)
    
    def test_dataloader(self):
        num_tasks=self.trainer.world_size
        global_rank = self.trainer.global_rank
        distributed_sampler = DistributedSampler(self.val_dataset, num_replicas=num_tasks, rank=global_rank)
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size,collate_fn=self.collate_fn, num_workers=4,sampler=distributed_sampler)
    
    def collate_fn(self, batch):
        videos = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # B, L, C, T, H, W
        return torch.stack(videos), labels
    
    def cal_weight(self,labels):
        label=labels.split('&')
        label1=label[0]
        label2=label[1]
        num_label1=self.data_distribution[label1]
        num_label2=self.data_distribution[label2]
        num_harmonic_mean=2/(1/num_label1+1/num_label2)
        # return weights
        return 1.0/num_harmonic_mean
    
    def apply_weight(self,dataset):
        weights=[]
        for idx,label in enumerate(dataset.labels):
            # print the complete percentage of the dataset
            print(f'Percentage of the dataset processed: {idx/len(dataset)*100}%')
            weighti=self.cal_weight(label)
            weights.append(weighti)
        return torch.tensor(weights)
    
    def load_weights(self):
        weight_path='/tsukimi/datasets/Chiba/baseline/weights.pkl'
        # if file exists, load the weights
        if os.path.exists(weight_path):
            with open(weight_path, 'rb') as f:
                weights=pickle.load(f)
        else:
            weights=self.apply_weight(self.train_dataset)
            # save the weights
            with open(weight_path, 'wb') as f:
                pickle.dump(weights,f)
        return weights
        