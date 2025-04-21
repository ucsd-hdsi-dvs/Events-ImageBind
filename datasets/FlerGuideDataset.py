# import os
# from typing import Optional, Callable

# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import models.imagebind_model as model_mod
from torchvision import transforms
import pickle as pkl
# import os.path as op
# import numpy as np
from PIL import Image
import pickle as pkl
# import json
# import cv2
# import random
# from datasets.utils.events_utils import gen_discretized_event_volume
# from data import load_and_transform_text_no_device

# from tqdm import tqdm
# from numpy_groupies import aggregate


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


class FlerGuide(Dataset):
    def __init__ (self, mode, transform=None):

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

        
        with open("/eastdata/datasets/sd-data/fler-guide/train_paths.pkl", 'rb') as f:
            train_paths = pkl.load(f)
        
        with open("/eastdata/datasets/sd-data/fler-guide/val_paths.pkl", 'rb') as f:
            test_paths = pkl.load(f)

        if mode == 'train':
            self.data_root, self.frame0_root, self.frame1_root = train_paths['fler'], train_paths['gray0'], train_paths['gray1']
        elif mode == 'test':
            self.data_root, self.frame0_root, self.frame1_root = test_paths['fler'], test_paths['gray0'], test_paths['gray1']

    def __len__(self):
        return len(self.data_root)
    
    def __getitem__(self, idx):
        data_path = self.data_root[idx]
        frame0_path = self.frame0_root[idx]
        frame1_path = self.frame1_root[idx]

        # read a png file
        FLER = Image.open(data_path).convert('RGB')
        frame0 = Image.open(frame0_path).convert('L').convert('RGB')
        frame1 = Image.open(frame1_path).convert('L').convert('RGB')
        
        FLER = self.transform(FLER)
        frame0 = self.transform(frame0)
        frame1 = self.transform(frame1)
        # stack frame0 and frame1
        frame = torch.stack((frame0, frame1), dim=1) 
        print('frame.shape', frame.shape)
        
        # print('rgb, voxel', rgb.shape, voxel.shape)
        return frame, model_mod.ModalityType.VISION, FLER, model_mod.ModalityType.EVENT


