from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.utils.data
from visualDet3D.utils.utils import alpha2theta_3d, theta2alpha_3d
from visualDet3D.data.kitti.kittidata import KittiData, KittiObj, KittiCalib
from visualDet3D.data.pipeline import build_augmentator
import os
import pickle
import numpy as np
from copy import deepcopy
from visualDet3D.networks.utils import BBox3dProjector
from visualDet3D.networks.utils.registry import DATASET_DICT
import sys
from matplotlib import pyplot as plt
ros_py_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if sys.version_info > (3, 0) and ros_py_path in sys.path:
    #Python 3, compatible with a naive ros environment
    sys.path.remove(ros_py_path)
    import cv2
    sys.path.append(ros_py_path)
else:
    #Python 2
    import cv2

@DATASET_DICT.register_module
class KittiMonoDataset(torch.utils.data.Dataset):
    """Some Information about KittiDataset"""
    def __init__(self, cfg, split='training'):
        super(KittiMonoDataset, self).__init__()
        preprocessed_path   = cfg.path.preprocessed_path
        obj_types           = cfg.obj_types
        is_train = (split == 'training')

        imdb_file_path = os.path.join(preprocessed_path, split, 'imdb.pkl')
        self.imdb = pickle.load(open(imdb_file_path, 'rb')) # list of kittiData
        self.output_dict = {
                "calib": False,
                "image": True,
                "label": False,
                "velodyne": False
            }
        if is_train:
            self.transform = build_augmentator(cfg.data.train_augmentation)
        else:
            self.transform = build_augmentator(cfg.data.test_augmentation)
        self.projector = BBox3dProjector()
        self.is_train = is_train
        self.obj_types = obj_types
        self.use_right_image = getattr(cfg.data, 'use_right_image', True)
        self.is_reproject = getattr(cfg.data, 'is_reproject', True) # if reproject 2d

    def _reproject(self, P2:np.ndarray, transformed_label:List[KittiObj]) -> Tuple[List[KittiObj], np.ndarray]:
        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        for obj in transformed_label:
            obj.alpha = theta2alpha_3d(obj.ry, obj.x, obj.z, P2)
        bbox3d_origin = torch.tensor([[obj.x, obj.y - 0.5 * obj.h, obj.z, obj.w, obj.h, obj.l, obj.alpha] for obj in transformed_label], dtype=torch.float32)
        abs_corner, homo_corner, _ = self.projector(bbox3d_origin, bbox3d_origin.new(P2))
        for i, obj in enumerate(transformed_label):
            extended_center = np.array([obj.x, obj.y - 0.5 * obj.h, obj.z, 1])[:, np.newaxis] #[4, 1]
            extended_bottom = np.array([obj.x, obj.y, obj.z, 1])[:, np.newaxis] #[4, 1]
            image_center = (P2 @ extended_center)[:, 0] #[3]
            image_center[0:2] /= image_center[2]

            image_bottom = (P2 @ extended_bottom)[:, 0] #[3]
            image_bottom[0:2] /= image_bottom[2]
            
            bbox3d_state[i] = np.concatenate([image_center,
                                                [obj.w, obj.h, obj.l, obj.alpha]]) #[7]

        max_xy, _= homo_corner[:, :, 0:2].max(dim = 1)  # [N,2]
        min_xy, _= homo_corner[:, :, 0:2].min(dim = 1)  # [N,2]

        result = torch.cat([min_xy, max_xy], dim=-1) #[:, 4]

        bbox2d = result.cpu().numpy()

        if self.is_reproject:
            for i in range(len(transformed_label)):
                transformed_label[i].bbox_l = bbox2d[i, 0]
                transformed_label[i].bbox_t = bbox2d[i, 1]
                transformed_label[i].bbox_r = bbox2d[i, 2]
                transformed_label[i].bbox_b = bbox2d[i, 3]
        
        return transformed_label, bbox3d_state


    def __getitem__(self, index):
        kitti_data = self.imdb[index % len(self.imdb)]
        # The calib and label has been preloaded to minimize the time in each indexing
        if index >= len(self.imdb):
            kitti_data.output_dict = {
                "calib": True,
                "image": False,
                "image_3":True,
                "label": False,
                "velodyne": False
            }
            calib, _, image, _, _ = kitti_data.read_data()
            calib.P2 = calib.P3 # a workaround to use P3 for right camera images. 3D bboxes are the same(cx, cy, z, w, h, l, alpha)
        else:
            kitti_data.output_dict = self.output_dict
            _, image, _, _ = kitti_data.read_data()
            calib = kitti_data.calib
        calib.image_shape = image.shape
        label = kitti_data.label # label: list of kittiObj
        label = []
        for obj in kitti_data.label:
            if obj.type in self.obj_types:
                label.append(obj)
        transformed_image, transformed_P2, transformed_label = self.transform(image, p2=deepcopy(calib.P2), labels=deepcopy(label))
        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        if len(transformed_label) > 0:
            transformed_label, bbox3d_state = self._reproject(transformed_P2, transformed_label)

        bbox2d = np.array([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in transformed_label])
        
        output_dict = {'calib': transformed_P2,
                       'image': transformed_image,
                       'label': [obj.type for obj in transformed_label], 
                       'bbox2d': bbox2d, #[N, 4] [x1, y1, x2, y2]
                       'bbox3d': bbox3d_state, 
                       'original_shape':image.shape,
                       'original_P':calib.P2.copy()}
        return output_dict

    def __len__(self):
        if self.is_train and self.use_right_image:
            return len(self.imdb) * 2
        else:
            return len(self.imdb)

    @staticmethod
    def collate_fn(batch):
        rgb_images = np.array([item["image"] for item in batch])#[batch, H, W, 3]
        rgb_images = rgb_images.transpose([0, 3, 1, 2])

        calib = [item["calib"] for item in batch]
        label = [item['label'] for item in batch]
        bbox2ds = [item['bbox2d'] for item in batch]
        bbox3ds = [item['bbox3d'] for item in batch]
        return torch.from_numpy(rgb_images).float(), torch.tensor(calib).float(), label, bbox2ds, bbox3ds

@DATASET_DICT.register_module
class NuscMonoDataset(KittiMonoDataset):
    def __len__(self):
        return len(self.imdb)

@DATASET_DICT.register_module
class KittiMonoTestDataset(KittiMonoDataset):
    def __init__(self, cfg, split='test'):
        preprocessed_path   = cfg.path.preprocessed_path
        obj_types           = cfg.obj_types
        super(KittiMonoTestDataset, self).__init__(cfg, 'test')
        is_train = (split == 'training')
        imdb_file_path = os.path.join(preprocessed_path, 'test', 'imdb.pkl')
        self.imdb = pickle.load(open(imdb_file_path, 'rb')) # list of kittiData
        self.output_dict = {
                "calib": False,
                "image": True,
                "label": False,
                "velodyne": False
            }

    def __getitem__(self, index):
        kitti_data = self.imdb[index % len(self.imdb)]
        kitti_data.output_dict = self.output_dict
        _, image, _, _ = kitti_data.read_data()
        calib = kitti_data.calib
        calib.image_shape = image.shape
        transformed_image, transformed_P2 = self.transform(
            image, p2=deepcopy(calib.P2))

        output_dict = {'calib': transformed_P2,
                       'image': transformed_image,
                       'original_shape':image.shape,
                       'original_P':calib.P2.copy()}
        return output_dict

    @staticmethod
    def collate_fn(batch):
        rgb_images = np.array([item["image"]
                               for item in batch])  # [batch, H, W, 3]
        rgb_images = rgb_images.transpose([0, 3, 1, 2])

        calib = [item["calib"] for item in batch]
        return torch.from_numpy(rgb_images).float(), calib 
