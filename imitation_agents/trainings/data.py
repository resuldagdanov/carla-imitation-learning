import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import lidar_to_histogram_features
from skimage import io


class DatasetLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.subfolder_paths = []

        subfolders = ["rgb_front_60", "rgb_right_45", "rgb_left_45", "measurements", "lidar"]

        for subfolder in subfolders:
            self.subfolder_paths.append(os.path.join(self.root_dir, subfolder))

        self.len = len(os.listdir(self.subfolder_paths[-1]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        front_img_name = os.path.join(self.subfolder_paths[0],  "%04i.png" % idx)
        right_img_name = os.path.join(self.subfolder_paths[1],  "%04i.png" % idx)
        left_img_name = os.path.join(self.subfolder_paths[2],  "%04i.png" % idx)
        meas_name = os.path.join(self.subfolder_paths[3],  "%04i.json" % idx)
        lidar_name = os.path.join(self.subfolder_paths[4],  "%04i.npy" % idx)

        front_image = io.imread(front_img_name)
        right_image = io.imread(right_img_name)
        left_image = io.imread(left_img_name)

        front_image = np.array(front_image.transpose((2, 0, 1)), np.float32)
        right_image = np.array(right_image.transpose((2, 0, 1)), np.float32)
        left_image = np.array(left_image.transpose((2, 0, 1)), np.float32)

        # normalize image
        front_image = front_image / 255
        right_image = right_image / 255
        left_image = left_image / 255

        # lidar: XYZI
        lidar_unprocessed = np.load(lidar_name)[...,:3]
        
        # convert lidar point cloud to image histogram
        lidar_processed = lidar_to_histogram_features(lidar_unprocessed, crop=256)

        with open(meas_name, 'r') as f:
            meas_json = json.load(f) 

        # global gps position of the ego vehicle
        ego_x = meas_json['x']
        ego_y = meas_json['y']

        # ego vehicle heading angle
        ego_theta = meas_json['theta']

        # far node
        x_command = meas_json['x_command']
        y_command = meas_json['y_command']

        # rotation matrix
        R = np.array([
            [np.cos(np.pi/2 + ego_theta), -np.sin(np.pi/2 + ego_theta)],
            [np.sin(np.pi/2 + ego_theta),  np.cos(np.pi/2 + ego_theta)]
            ])

        # convert far nodes to relative local waypoints
        local_command_point = np.array([x_command - ego_x, y_command - ego_y])
        local_command_point = R.T.dot(local_command_point)

        sample = {
            "fronts": front_image,
            "rights": right_image,
            "lefts": left_image,
            "lidars": lidar_processed,
            "velocity": np.array([meas_json['speed']], np.float32),
            'control': np.array([meas_json['throttle'], meas_json['steer'], meas_json['brake']], np.float32),
            "target_point": tuple(local_command_point)
        }
        return sample
