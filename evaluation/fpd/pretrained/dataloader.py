from torch.utils.data import Dataset
import numpy as np
import glob
import json
import os

def pc_normalize_axiswise(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    longest_axis_range = np.max(pc, axis=0) - np.min(pc, axis=0)
    longest_axis = np.argmax(longest_axis_range).item()

    # Find the min and max values along this longest axis
    min_val = np.min(pc[:, longest_axis])
    max_val = np.max(pc[:, longest_axis])

    # Normalize the point cloud based on the longest axis
    pc = (pc - min_val) / (max_val - min_val)
    return pc


class NuscenesGeneratedObjectsDataLoader(Dataset):
    def __init__(self, root, split, real_or_generated, num_points, input_channels, object_class, logen=True, multiclass_logen=False):
        super().__init__()
        paths = glob.glob(f'{root}/{object_class}/**')
        print(f'{root}/{object_class}/**')
        self.object_name = 'generated_0' if real_or_generated=='generated' else 'original_0'
        self.npoints = num_points
        self.input_channels = input_channels
        self.dirs = [path for path in paths if path.split('/')[-3]]

    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, index):

        curr_dir = self.dirs[index]
        object_data = np.loadtxt(f'{curr_dir}/{self.object_name}.txt', dtype=np.float32)
        label = 1
        points = object_data[:, :3]
        points = pc_normalize_axiswise(points)

        if self.input_channels == 4:
            intensity = object_data[:, 3]
            data_log_transformed = np.log1p(intensity)
            max_intensity = np.log1p(255.0)
            intensity = data_log_transformed / max_intensity
            points = np.column_stack((points, intensity))
        return points, label, curr_dir
