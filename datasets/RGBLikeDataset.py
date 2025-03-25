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
    def __init__ (self, data_root, mode, transform=None, frame_size=(260,346), num_bins=6):
        with open('/eastdata/datasets/MVSEC/data_paths_slices.pkl', 'rb') as f:
            paths_pack = pkl.load(f)
        self.data_root = data_root + 'event_chunks_processed_train/'
        train_paths, test_paths = train_test_split(paths_pack['train'], test_size=0.2, random_state=42)
        if mode == 'train':
            self.data_paths = train_paths
        elif mode == 'test':
            self.data_paths = test_paths
            
        self.transform = transform
        self.frame_size = frame_size
        self.num_bins = num_bins
        self.event_frame_normalize = transforms.Compose([
                resize_pad,
                transforms.Normalize([0.127, 0.143, 0.267], [0.581, 0.610, 1.05])])
        
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_path = op.join(self.data_root, self.data_paths[idx])
        with open(data_path, 'rb') as f:
            data_packet = pkl.load(f)

        # Unpack data
        events = data_packet['events']
        voxel = gen_discretized_event_volume(events, [self.num_bins, *self.frame_size])

        image_units=[]
        for i in range(len(data_packet['frames'])):
            image=data_packet['frames'][i]
            # convert to 3 channels
            image=np.repeat(image[...,None],3,axis=2).transpose(2,0,1)
            image=torch.from_numpy(image).float()/255
            image=self.event_frame_normalize(image)
            image_units.append(image)
        
        image_units=torch.stack(image_units) 
        image_units=torch.stack([image_units[:-1],image_units[1:]],dim=2) 
        
        return image_units[0], model_mod.ModalityType.VISION, voxel, model_mod.ModalityType.EVENT




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



# def convert_path(npz_path):

#     new_base = "/eastdata/datasets/Caltech101/101_ObjectCategories"
#     parts = npz_path.split('/')

#     category = parts[-2]  
#     file_name = parts[-1] 
    
#     new_file_name = file_name.replace('.npz', '.jpg')
#     new_path = os.path.join(new_base, category, new_file_name)
    
#     return new_path

def process_file_paths(file_path):
    """
    Reads the file at `file_path`, processes the bin and jpg file paths,
    converts the bin paths to npz paths, and updates the jpg paths to a new format.

    Args:
    - file_path (str): Path to the input file containing bin and jpg file paths.

    Returns:
    - List: A list containing two lists: 
        - The first list contains converted npz file paths.
        - The second list contains converted jpg file paths.
    """
    # Read and process lines
    with open(file_path, 'r') as file:
        data = [line.strip().split('\t') for line in file]

    # Extract bin and jpg file paths
    bin_files = [item[0] for item in data]  # First column (bin paths)
    jpg_files = [item[1] for item in data]  # Second column (jpg paths)

    # Convert bin paths to npz paths
    npz_files = [
        f"/eastdata/datasets/imagebind_ncaltech101/March_1/{bin_path.split('/')[1]}/image_{bin_path.split('_')[-1]}"
        .replace('.bin', '.png')
        for bin_path in bin_files
    ]

    # Convert jpg paths to the new format
    jpg_files_converted = [
        f"/eastdata/datasets/Caltech101/101_ObjectCategories/{jpg_path.split('/')[1]}/{jpg_path.split('/')[-1]}"
        for jpg_path in jpg_files
    ]

    return [npz_files, jpg_files_converted]


def background_filter(events, frame_size=(260, 346), dt=70000):

    """ Filter out the background events that are actually noise.
        Works by looking at the time difference between an event and the temporally closest event in the past and in the future at the same xy pixel coordinates. If both of these delta times are smaller than a predefined threshold dt, treat this event as noise and remove it.
    Args:
        events: ndarray, shape:(ta, 4), the events stream
        frame_size: tuple/list, (h, w), the frame size
        dt: int, threshold of 'long time' in us.
    Returns:
        events_filtered: ndarray, shape: (?, 4), filtered events.
    """
    h, w = frame_size
    ta = events.shape[0]
    
    # Arrays to store time deltas
    deltaT_past = np.full(ta, np.inf)
    deltaT_future = np.full(ta, np.inf)
    
    # Dictionary to track last and next timestamps for each pixel
    lastTimesMap = np.full((w, h), -np.inf)
    nextTimesMap = np.full((w, h), np.inf)
    
    # First pass: Calculate past deltas
    for i in range(ta):
        ts, xs, ys, ps = events[i]
        if lastTimesMap[(xs, ys)] != -np.inf:
            deltaT_past[i] = ts - lastTimesMap[(xs, ys)]
        lastTimesMap[max(0, xs-2):min(w, xs+2), max(0, ys-2):min(h, ys+2)] = ts
    
    # Second pass: Calculate future deltas
    for i in reversed(range(ta)):
        ts, xs, ys, ps = events[i]
        if nextTimesMap[(xs, ys)] != np.inf:
            deltaT_future[i] = nextTimesMap[(xs, ys)] - ts
        nextTimesMap[max(0, xs-2):min(w, xs+2), max(0, ys-2):min(h, ys+2)] = ts
    
    # Filter events based on both past and future deltas
    valid_indices = (deltaT_past <=  dt) | (deltaT_future <= dt)
    events_filtered = events[valid_indices]
    
    return events_filtered


def hot_pixel_filter(events, frame_size=(260, 346), thres_percentile=99):
    """ Filter out the Hot Pixels.
        Hot pixels are defined as the pixels that record a number of event
        bigger than threventhotpixel.
    Args:
        events: ndarray, shape:(ta, 4), the events stream
        frame_size: tuple/list, (h, w), the frame size
        thres_percentile: float, 0~100, threshold percentile of hot pixel.
    Returns:
        events_filtered: ndarray, shape: (?, 4), filtered events.
    """
    h, w = frame_size

    hotpixelarray = aggregate(events[:, 1:3].T, np.ones_like(
        events[:, 0]), func='sum', size=(w, h), fill_value=0)
    threventhotpixel = np.percentile(hotpixelarray.flatten(), thres_percentile)
    selindexarray = hotpixelarray >= threventhotpixel
    [hpx, hpy] = np.nonzero(selindexarray.astype(int))
    fs = np.array((hpx, hpy)).T

    events_filtered = events[~(events[:, 1:3] == fs[:, None]).all(-1).any(axis=0)]
    return events_filtered