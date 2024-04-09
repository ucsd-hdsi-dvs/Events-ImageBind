import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from models.imagebind_model import ModalityType
import data
from torchvision import transforms
import pickle as pkl
import os.path as op
import numpy as np
import cv2

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


class EventDataset(Dataset):
    
    def __init__(self,mode, data_dir,partial_dataset=1, transform=None, device: str = 'cpu',seq_len=2,frame_size=(260, 346),path='data/paths.pkl'):
        
        self.mode=mode
        self.data_root = data_dir
        self.transform = transform
        self.seq_len = seq_len
        self.frame_size = frame_size
        
        self.frame_normalize = transforms.Compose([
                transforms.Normalize([0.153, 0.153, 0.153], [0.165, 0.165, 0.165])])
        
        
        with open(path, 'rb') as f:
            self.paths_pack = pkl.load(f)
            
        if mode == 'train':
            self.data_paths = self.paths_pack['train']
        elif mode == 'val':
            self.data_paths = self.paths_pack['val']
        elif mode == 'test':
            self.data_paths = self.paths_pack['test']
        

    def __len__(self):
        return int(self.partial_dataset*len(self.data_paths))

    def __getitem__(self, idx):
        data_path = op.join(self.data_root, self.data_paths[idx])
        with open(data_path, 'rb') as f:
            data_packet = pkl.load(f)

        events = data_packet['events'][0]
        image_list = data_packet['images'] # 16, 260, 346
        #convert to grayscale image
        image_list=[cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB) for image in image_list]
        image_list = np.stack(image_list, axis=0)
        image_units = torch.from_numpy(image_list).float() / 255 # 2, 260, 346, 3
        image_units=image_units.permute(0, 3, 1, 2) # 2,3,260,346
        image_units = self.frame_normalize(image_units)
        
        # replace polarity 0 with -1
        events['polarity'][events['polarity']==0]=-1
        
        # generate positive and negative channel
        events_positive=events[events['polarity']==1]
        events_negative=events[events['polarity']==-1]
        events_positive_frame=events_to_image_torch(events_positive['x'],events_positive['y'],events_positive['polarity'])
        events_negative_frame=events_to_image_torch(events_negative['x'],events_negative['y'],events_negative['polarity'])
        
        # stack (positive,negative,positive+negative) channel
        event_frame=torch.stack([events_positive_frame,events_negative_frame,events_positive_frame+events_negative_frame],dim=0)
        
            
        return {
            'image_units': image_units, # [2, 3, H, W]
            'event_frame': event_frame, # [3, H, W]
            'data_path': data_path
        }
    
    