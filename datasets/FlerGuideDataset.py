import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import models.imagebind_model as model_mod
from torchvision import transforms
import pickle as pkl
import os.path as op
import numpy as np
from PIL import Image
import json
import cv2
import random
from datasets.utils.events_utils import gen_discretized_event_volume
from data import load_and_transform_text_no_device

from tqdm import tqdm
from numpy_groupies import aggregate


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


class RGBLikeCaltech(Dataset):
    def __init__ (self, data_root, mode, transform=None, frame_size=(224,224), num_bins=20):
        self.frame_size = frame_size
        self.num_bins = num_bins
        self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        224, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )

        # with open(data_root, 'r') as f:
        #     paths = json.load(f)
        # train_paths, test_paths = train_test_split(paths, test_size=0.2, random_state=42)
        

        eventbind_train = "/eastdata/datasets/N-Caltech101/Caltech101_train.txt"
        eventbind_val = "/eastdata/datasets/N-Caltech101/Caltech101_val.txt"

        
        train_paths, test_paths = process_file_paths(eventbind_train), process_file_paths(eventbind_val)
        
        # Open the file and load its contents
        with open('/eastdata/datasets/N-Caltech101/Caltech101_classnames.json', 'r') as file:
            self.classnames_dict= json.load(file)
            
        if mode == 'train':
            self.data_root, self.frame_root = train_paths[0], train_paths[1]
        elif mode == 'test':
            self.data_root, self.frame_root = test_paths[0], test_paths[1]

    def __len__(self):
        return len(self.data_root)
    
    def __getitem__(self, idx):
        data_path = self.data_root[idx]
        rgb_path = self.frame_root[idx]
        label_str=data_path.split('/')[-2]
        label_str= 'A sketch image of a ' + label_str
        label_str=load_and_transform_text_no_device([label_str])
        # print('shape of label_str', label_str.shape)
        label_str=label_str.squeeze(0)
        # label_idx = int(self.classnames_dict[label_str])
        
        # read a png file
        FLER = Image.open(data_path).convert('RGB')
        rgb = Image.open(rgb_path).convert('RGB')
        
        rgb = self.transform(rgb)
        # rgb = rgb.unsqueeze(1)
        # rgb = rgb.repeat(1, 2, 1, 1)
        
        
        FLER = self.transform(FLER)
        
        # print('rgb, voxel', rgb.shape, voxel.shape)
        return rgb, model_mod.ModalityType.VISION, FLER, model_mod.ModalityType.EVENT, label_str, model_mod.ModalityType.TEXT


