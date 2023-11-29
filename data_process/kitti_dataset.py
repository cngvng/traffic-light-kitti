import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import sys
import os
import math
from builtins import int

import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import icecream as ic

sys.path.append("/workspaces/PheNet-Traffic_light/")

from data_process.kitti_data_utils import gen_hm_radius, compute_radius, Calibration
import data_process.transformation as transformation
import config.kitti_config as cnf

class KittiDataset(Dataset):
    def __init__(self, configs, mode='train',  stage='raw'):
        self.dataset_dir = configs.dataset_dir
        self.input_size = configs.input_size
        self.hm_size = configs.hm_size

        self.num_classes = configs.num_classes
        self.max_objects = configs.max_objects

        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        sub_folder = 'testing' if self.is_test else 'training'

        self.stage = stage

        self.image_dir = os.path.join(self.dataset_dir, sub_folder, "image_2")
        self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "velodyne")
        self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")
        self.label_dir = os.path.join(self.dataset_dir, sub_folder, f"label_traffic_sign_{self.stage}")
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', '{}.txt'.format(mode))
        self.sample_id_list = [int(x.strip()) for x in open(split_txt_path).readlines()]

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_img_only(index)
        else:
            return self.load_img_with_targets(index)

    def load_img_only(self, index):
        """Load only image for the testing phase"""
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        metadatas = {
            'img_path': img_path,
        }

        return metadatas, img_rgb

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""
        sample_id = int(self.sample_id_list[index])
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))
        lidarData = self.get_lidar(sample_id)
        calib = self.get_calib(sample_id)
        labels, has_labels = self.get_label(sample_id)



        metadatas = {
            'img_path': img_path,
        }

        return metadatas,  labels

    def get_image(self, idx):
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return img_path, img

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return Calibration(calib_file)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
        # assert os.path.isfile(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_label(self, idx):
        labels = []
        sample_id = int(self.sample_id_list[idx])
        label_path = os.path.join(self.label_dir, '{:06d}.txt'.format(sample_id))
        for line in open(label_path, 'r'):
            line = line.rstrip()
            line_parts = line.split(' ')
            gt_id = line_parts[0]
            # location (x, y, z) in camera coord
            x, y, z = float(line_parts[1]), float(line_parts[2]), float(line_parts[3])
            object_label = [gt_id, x, y, z]
            labels.append(object_label)
        if len(labels) == 0:
            labels = np.zeros((1, 4), dtype=np.float32)
            has_labels = False
        else:
            labels = np.array(labels, dtype=np.float32)
            has_labels = True

        return labels, has_labels

    def build_targets(self, labels, hflipped):
        minX = cnf.boundary['minX']
        maxX = cnf.boundary['maxX']
        minY = cnf.boundary['minY']
        maxY = cnf.boundary['maxY']
        minZ = cnf.boundary['minZ']
        maxZ = cnf.boundary['maxZ']

        num_objects = min(len(labels), self.max_objects)
        hm_l, hm_w = self.hm_size

        hm_main_center = np.zeros((self.num_classes, hm_l, hm_w), dtype=np.float32)
        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        z_coor = np.zeros((self.max_objects, 1), dtype=np.float32)
        dimension = np.zeros((self.max_objects, 3), dtype=np.float32)

        indices_center = np.zeros((self.max_objects), dtype=np.int64)
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8)

        for k in range(num_objects):
            gt_id, x, y, z = labels[k]
            gt_id = int(gt_id)

            bbox_l = 1 / cnf.bound_size_x * hm_l
            bbox_w = 1 / cnf.bound_size_y * hm_w
            radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
            
            center_y = (x - minX) / cnf.bound_size_x * hm_l 
            center_x = (y - minY) / cnf.bound_size_y * hm_w
            center = np.array([center_x, center_y], dtype=np.float32)

            if hflipped:
                center[0] - hm_w - center[0] - 1
            
            center_int = center.astype(np.int32)
            
            gen_hm_radius(hm_main_center[gt_id], center, radius)

            cen_offset[k] = center - center_int

            z_coor[k] = z - minZ

            obj_mask[k] = 1
        
        targets = {
            'hm_cen': hm_main_center,
            'cen_offset': cen_offset,
            'z_coor': z_coor,
            'dim': dimension,
            'indices_center': indices_center,
            'obj_mask': obj_mask,
        }

        return targets

    def draw_img_with_label(self, index):
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        calib = self.get_calib(sample_id)
        labels, has_labels = self.get_label(index)

        return labels, img_rgb, img_path, calib


if __name__ == '__main__':
    from easydict import EasyDict as edict
    from data_process.transformation import OneOf, Random_Scaling, Random_Rotation, lidar_to_camera_box
    from utils.visualization_utils import  show_rgb_image_with_boxes

    configs = edict()
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.num_samples = None
    configs.input_size = (608, 608)
    configs.hm_size = (20, 20)
    configs.max_objects = 50
    configs.num_classes = 3
    configs.output_width = 608

    configs.dataset_dir = os.path.join(  'kitti')

    dataset = KittiDataset(configs, mode='train')

    print('\n\nPress n to see the next sample >>> Press Esc to quit...')
    for idx in range(len(dataset)):
        labels, img_rgb, img_path, calib = dataset.draw_img_with_label(idx)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_rgb = show_rgb_image_with_boxes(img_rgb, labels, calib)
        cv2.imshow("img", img_rgb)

        if cv2.waitKey(0) & 0xff == 27:
            break