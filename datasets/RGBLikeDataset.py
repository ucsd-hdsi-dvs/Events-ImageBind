import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from models.imagebind_model import ModalityType
from torchvision import transforms
import pickle as pkl
import os.path as op
import numpy as np
import cv2
from models.imagebind_model import ModalityType

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


def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(260, 346), clip_out_of_range=False,
        interpolation=None, padding=True, default=0):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
    @param xs Tensor of x coords of events
    @param ys Tensor of y coords of events
    @param ps Tensor of event polarities/weights
    @param device The device on which the image is. If none, set to events device
    @param sensor_size The size of the image sensor/output image
    @param clip_out_of_range If the events go beyond the desired image size,
       clip the events to fit into the image
    @param interpolation Which interpolation to use. Options=None,'bilinear'
    @param padding If bilinear interpolation, allow padding the image by 1 to allow events to fit:
    @returns Event image from the events
    """
    xs=torch.tensor(xs.copy(), dtype=torch.long)
    ys = torch.tensor(ys.copy(), dtype=torch.long)
    ps = torch.tensor(ps.copy(),dtype=torch.float32)
    
    if device is None:
        device = xs.device
    img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = (torch.ones(img_size)*default).to(device)
    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    try:
        mask = mask.long().to(device)
        xs, ys = xs*mask, ys*mask
        img.index_put_((ys, xs), ps, accumulate=True)
    except Exception as e:
        print("Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
            ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
        raise e
    return img

class RGBLikeDataset(Dataset):
    
    def __init__(self,mode, data_dir, transform=None,seq_len=2,frame_size=(260, 346),random_seed=42):
        self.mode=mode
        self.data_root = data_dir
        self.transform = transform
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.frame_normalize = transforms.Compose([
                resize_pad,
                transforms.Normalize([0.153, 0.153, 0.153], [0.165, 0.165, 0.165])])
        
        self.event_frame_normalize = transforms.Compose([
                resize_pad,
                transforms.Normalize([0.127, 0.143, 0.267], [0.581, 0.610, 1.05])])
        
        self.paths = []
        for file in os.listdir(data_dir):
            if file.endswith('.pkl'):
                self.paths.append(os.path.join(data_dir, file))
        
        train_paths, test_paths = train_test_split(self.paths, train_size=0.8, random_state=random_seed)
        if self.mode == 'train':
            self.data_paths = train_paths
        elif self.mode == 'test':
            self.data_paths = test_paths
        else:
            raise ValueError(f"Invalid mode argument. Expected 'train' or 'test', got {self.mode}")
        
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        with open(data_path, 'rb') as f:
            data_packet = pkl.load(f)
        
        
        image_units=[]
        for i in range(len(data_packet['images'])):
            image=data_packet['images'][i]
            # convert to 3 channels
            image=np.repeat(image[...,None],3,axis=2).transpose(2,0,1)
            image=torch.from_numpy(image).float()/255
            image=self.frame_normalize(image)
            image_units.append(image)
        
        image_units=torch.stack(image_units) # 2, 3, 224, 224
        image_units=torch.stack([image_units[:-1],image_units[1:]],dim=2) # 1, 3,2, 224, 224
        # image_units = np.stack([image_units[:-1], image_units[1:]], axis=1) 
        
        events=data_packet['events'] # (x,y,t,p)
        # convert events to event frame
        events['polarity'][events['polarity']==0]=-1
        events_positive=events[events['polarity']==1]
        events_negative=events[events['polarity']==-1]
        events_positive_frame=events_to_image_torch(events_positive['x'],events_positive['y'],events_positive['polarity'])
        events_negative_frame=events_to_image_torch(events_negative['x'],events_negative['y'],events_negative['polarity'])
        # abs negative channel
        events_negative_frame=torch.abs(events_negative_frame)
        events_sum_frame=events_positive_frame+events_negative_frame
        
        event_frame=torch.stack([events_positive_frame,events_negative_frame,events_sum_frame],dim=0)
        # event_frame=resize_pad(event_frame)
        event_frame=self.event_frame_normalize(event_frame)
        
        return image_units[0],ModalityType.VISION, event_frame, ModalityType.EVENT
        