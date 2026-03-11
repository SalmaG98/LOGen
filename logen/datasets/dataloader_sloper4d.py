import os
import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import Dataset
import pickle
import numpy as np
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box, Quaternion
import open3d as o3d
from logen.modules.three_d_helpers import cartesian_to_cylindrical, angle_difference
from logen.modules.class_mapping import class_mapping

dataset_path = "/home/sgalaaou/scania/datasets_scania/xixihahaya___SLOPER4D-dataset"

def scale_intensity(data):
    data_log_transformed = np.log1p(data) 
    max_intensity = np.log1p(255.0)
    return data_log_transformed / max_intensity
    
class Sloper4DObjectsSet(Dataset):
    def __init__(self, 
                data_dir,
                split, 
                volume_expansion=1., 
                input_channels=3,
                excluded_tokens=None,
                permutation=[],
                n_samples=-1,
                ):
        super().__init__()
        with open(data_dir, 'rb') as f:
            self.data_index = pickle.load(f)[split]
        
        if isinstance(self.data_index, dict):
            if excluded_tokens != None:
                print(f'Before existing object filtering: {len(self.data_index)} objects')
                self.data_index = [value for key, value in self.data_index.items() if key not in excluded_tokens]
                print(f'After existing object filtering: {len(self.data_index)} objects')
            else:
                self.data_index = list(self.data_index.values())
    
        if len(permutation) > 0:
            print(f'Limiting dataset to {len(permutation)} samples')
            self.data_index = [self.data_index[i] for i in permutation]
            print(f'After limiting, length of dataset is {len(self.data_index)}')

        self.nr_data = len(self.data_index)
        self.volume_expansion = volume_expansion
        self.input_channels = input_channels
        self.n_samples = n_samples

    def __len__(self):
        if self.n_samples != -1:
            return self.n_samples
        else:
            return self.nr_data
    
    def __getitem__(self, index):
        object_json = self.data_index[index]
        
        class_name = 'human'
        
        points = object_json['world2lidar'] @ np.hstack([object_json['human_points'], np.ones((object_json['human_points'].shape[0],1))]).T
        
        center = np.array(object_json['smpl_lidar_trans'])
        size = np.array(object_json['smpl_betas'])

        orientation = object_json['smpl_global_pose']
        orientation[:3] = object_json['smpl_lidar_orient']

        object_points = points.T[:,:3] 
        num_points = object_points.shape[0]
        padding_mask = torch.zeros((object_points.shape[0]))
        # SG(TODO): transform points in a local frame
        center = cartesian_to_cylindrical(center).squeeze(0)

        class_label = torch.tensor(class_mapping[class_name])

        token = object_json['pc_path'].split('/')[-4]+'-'+object_json['pc_path'].split('/')[-1]

        return [object_points, center, torch.from_numpy(size), orientation, num_points, class_label, padding_mask, token]