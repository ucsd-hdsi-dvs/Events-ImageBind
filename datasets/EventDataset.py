import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from models.imagebind_model import ModalityType
import data
from torchvision import transforms
import pickle as pkl
import os.path as op

class EventDataset(Dataset):
    
    def __init__(self,mode, data_dir,partial_dataset=1, transform=None, device: str = 'cpu',seq_len=2,frame_size=(260, 346)):
        
        self.mode=mode
        self.data_root = data_dir
        self.transform = transform
        self.seq_len = seq_len
        self.frame_size = frame_size
        
        self.frame_normalize = transforms.Compose([
                transforms.Normalize([0.153, 0.153], [0.165, 0.165])])
        
        
        with open('', 'rb') as f:
            self.paths_pack = pkl.load(f)
            
        if mode == 'train':
            self.data_paths = self.paths_pack['train']
        elif mode == 'val':
            self.data_paths = self.paths_pack['val']
        elif mode == 'test':
            self.data_paths = self.paths_pack['test']
        
        raise NotImplementedError

    def __len__(self):
        return int(self.partial_dataset*len(self.data_paths))

    def __getitem__(self, idx):
        data_path = op.join(self.data_root, self.data_paths[idx])
        with open(data_path, 'rb') as f:
            data_packet = pkl.load(f)

        events = data_packet['events']
        # lfr = gen_log_frame_residual_batch(data_packet['images'])
        # lfr = torch.from_numpy(lfr).float()
        image_list = data_packet['images'] # 16, 260, 346
        #convert to grayscale image
        # image_list=[cv2.cvtColor(np.array(image),cv2.COLOR_BGR2GRAY) for image in image_list]
        image_units = np.stack([image_list[:-1], image_list[1:]], axis=1) # 16, 2, 260, 346

        # physical_att = physical_attention_batch_generation(events, image_units, self.phyatt_grid_size, advanced=self.advanced_physical_att, ceiling=self.ceiling_att)
        # physical_att = torch.from_numpy(physical_att).float().unsqueeze(1) # 16, 1, 260, 346

        image_units = torch.from_numpy(image_units).float() / 255
        if self.apply_image_grad:
            image_gradient_blur = get_batch_double_blurred_image_gradient(image_units[:, 0:1], image_units[:, 1:2])
            image_gradient_blur = image_gradient_blur / image_gradient_blur.max()
            image_units = self.frame_normalize(image_units)
            image_units = torch.cat([image_units, image_gradient_blur], dim=1)
        else:
            image_units = self.frame_normalize(image_units)
            
        # gyroscopes = torch.from_numpy(data_packet['gyroscopes']).float()
        # accelerometers = torch.from_numpy(data_packet['accelerometers']).float()
        # optical_flow = torch.from_numpy(data_packet['optical_flow']).float()
        # optical_flow = self.normalize_optical_flow(optical_flow)

        # physical_att = torch.from_numpy(data_packet['physical_att']).float().unsqueeze(1) # 16, 1, 260, 346

        #events_window=data_packet['events_window']
        # acc_flow = torch.from_numpy(data_packet['acc_flow']).float()

        # optical_flow = data_packet['optical_flow'].float().cpu()
        # acc_flow = torch.from_numpy(data_packet['acc_flow']).float()
        # acc_flow = self.normalize_flows(acc_flow)
        # # acc_flow = data_packet['acc_flow'].float()
        # flows = torch.cat([optical_flow, acc_flow], axis=1) #16, 4, 260, 346

        # start buuilding the torch for voxels based on events input
        voxels = []
        for i in range(len(events)):
            # time_voxel = structured_events_to_voxel_grid(events[i], num_bins=self.num_bins, width=346, height=260)
            time_voxel = gen_discretized_event_volume(events[i],
                                                    [self.num_bins*2,
                                                     self.frame_size[0],
                                                     self.frame_size[1]])

            # time_voxel = torch.from_numpy(time_voxel).float()
            voxels.append(time_voxel)
            #accumulate_voxel=structured_events_to_voxel_grid(events_window[i], num_bins=self.num_bins, width=346, height=260)
            #accumulate_voxel=torch.from_numpy(accumulate_voxel).float()
            #accumulate_voxels.append(accumulate_voxel)
            #print("The shape of time_voxel:",time_voxel.shape)
            
        voxels = torch.stack(voxels, dim=0)
        #accumulate_voxels = torch.stack(accumulate_voxels, dim=0)
        # imu = torch.cat([accelerometers, gyroscopes], axis=1)[1:]

        if 0 < self.seq_len < 16:
            # lfr = lfr[:self.seq_len]
            image_units = image_units[:self.seq_len]
            # flows = flows[:self.seq_len]
            voxels = voxels[:self.seq_len]
            # imu = imu[:self.seq_len]
            # physical_att = physical_att[:self.seq_len]

        if self.mode == 'train' and self.random_flip:
            image_units, voxels, imu = seq_random_flip(image_units, voxels, imu, self.flip_x_prob, self.flip_y_prob)
            
        return {
            'image_units': image_units, # [L, 2, H, W]
            # 'flows': flows, # [L, 4, H, W]
            'voxels': voxels, # [L, 2*num_bin, H, W]
            'imu': imu, # [L, 6]
            #'physical_att': physical_att, # [L, 1, H, W]
            # 'lfr': lfr, # [L, 1, H, W]
            'data_path': data_path
        }
    
    