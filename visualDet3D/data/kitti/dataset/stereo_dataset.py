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
import torch.utils.data
from visualDet3D.data.kitti.kittidata import KittiData, KittiObj, KittiCalib
from visualDet3D.data.pipeline import build_augmentator

import os
import pickle
import numpy as np
from copy import deepcopy
from visualDet3D.utils.utils import alpha2theta_3d, theta2alpha_3d, draw_3D_box
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
class KittiStereoDataset(torch.utils.data.Dataset):
    """Some Information about KittiDataset"""
    def __init__(self, cfg, split='training'):
        super(KittiStereoDataset, self).__init__()
        preprocessed_path   = cfg.path.preprocessed_path
        obj_types           = cfg.obj_types
        aug_cfg             = cfg.data.augmentation
        is_train = (split == 'training')
        imdb_file_path = os.path.join(preprocessed_path, split, 'imdb.pkl')
        self.imdb = pickle.load(open(imdb_file_path, 'rb')) # list of kittiData
        self.output_dict = {
                "calib": True,
                "image": True,
                "image_3":True,
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
        self.preprocessed_path = preprocessed_path

    def _reproject(self, P2:np.ndarray, transformed_label:List[KittiObj]) -> Tuple[List[KittiObj], np.ndarray]:
        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        if len(transformed_label) > 0:
            #for obj in transformed_label:
            #    obj.alpha = theta2alpha_3d(obj.ry, obj.x, obj.z, P2)
            bbox3d_origin = torch.tensor([[obj.x, obj.y - 0.5 * obj.h, obj.z, obj.w, obj.h, obj.l, obj.alpha] for obj in transformed_label], dtype=torch.float32)
            try:
                abs_corner, homo_corner, _ = self.projector.forward(bbox3d_origin, bbox3d_origin.new(P2))
            except:
                print('\n',bbox3d_origin.shape, len(transformed_label), len(label), label, transformed_label, bbox3d_origin)
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

            for i in range(len(transformed_label)):
                transformed_label[i].bbox_l = bbox2d[i, 0]
                transformed_label[i].bbox_t = bbox2d[i, 1]
                transformed_label[i].bbox_r = bbox2d[i, 2]
                transformed_label[i].bbox_b = bbox2d[i, 3]
        return transformed_label, bbox3d_state
        
    def __getitem__(self, index):
        kitti_data = self.imdb[index]
        # The calib and label has been preloaded to minimize the time in each indexing
        kitti_data.output_dict = self.output_dict
        calib, left_image, right_image, _, _ = kitti_data.read_data()
        calib.image_shape = left_image.shape
        label = []
        for obj in kitti_data.label:
            if obj.type in self.obj_types:
                label.append(obj)
        transformed_left_image, transformed_right_image, P2, P3, transformed_label = self.transform(
                left_image, right_image, deepcopy(calib.P2),deepcopy(calib.P3), deepcopy(label)
        )
        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        
        if len(transformed_label) > 0:
            transformed_label, bbox3d_state = self._reproject(P2, transformed_label)

        if self.is_train:
            if abs(P2[0, 3]) < abs(P3[0, 3]): # not mirrored or swaped, disparity should base on pointclouds projecting through P2
                disparity = cv2.imread(os.path.join(self.preprocessed_path, 'training', 'disp', "P2%06d.png" % index), -1)
            else: # mirrored and swap, disparity should base on pointclouds projecting through P3, and also mirrored
                disparity = cv2.imread(os.path.join(self.preprocessed_path, 'training', 'disp', "P3%06d.png" % index), -1)
                disparity = disparity[:, ::-1]
            disparity = disparity / 16.0
        else:
            disparity = None

        bbox2d = np.array([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in transformed_label])
        
        output_dict = {'calib': [P2, P3],
                       'image': [transformed_left_image, transformed_right_image],
                       'label': [obj.type for obj in transformed_label], 
                       'bbox2d': bbox2d, #[N, 4] [x1, y1, x2, y2]
                       'bbox3d': bbox3d_state,
                       'original_shape': calib.image_shape,
                       'disparity': disparity,
                       'original_P':calib.P2.copy()}
        return output_dict

    def __len__(self):
        return len(self.imdb)

    @staticmethod
    def collate_fn(batch):
        left_images = np.array([item["image"][0] for item in batch])#[batch, H, W, 3]
        left_images = left_images.transpose([0, 3, 1, 2])

        right_images = np.array([item["image"][1] for item in batch])#[batch, H, W, 3]
        right_images = right_images.transpose([0, 3, 1, 2])

        P2 = [item['calib'][0] for item in batch]
        P3 = [item['calib'][1] for item in batch]
        label = [item['label'] for item in batch]
        bbox2ds = [item['bbox2d'] for item in batch]
        bbox3ds = [item['bbox3d'] for item in batch]
        disparities = [item['disparity'] for item in batch]
        if disparities[0] is None:
            return torch.from_numpy(left_images).float(), torch.from_numpy(right_images).float(), torch.tensor(P2).float(), torch.tensor(P3).float(), label, bbox2ds, bbox3ds
        else:
            return torch.from_numpy(left_images).float(), torch.from_numpy(right_images).float(), torch.tensor(P2).float(), torch.tensor(P3).float(), label, bbox2ds, bbox3ds, torch.tensor(disparities).float()

@DATASET_DICT.register_module
class KittiStereoTestDataset(KittiStereoDataset):
    def __init__(self, cfg, split='test'):
        preprocessed_path   = cfg.path.preprocessed_path
        obj_types           = cfg.obj_types
        aug_cfg             = cfg.data.augmentation
        super(KittiStereoTestDataset, self).__init__(cfg, split)
        imdb_file_path = os.path.join(preprocessed_path, 'test', 'imdb.pkl')
        self.imdb = pickle.load(open(imdb_file_path, 'rb')) # list of kittiData
        self.output_dict = {
                "calib": True,
                "image": True,
                "image_3":True,
                "label": False,
                "velodyne": False
            }

    def __getitem__(self, index):
        kitti_data = self.imdb[index]
        # The calib and label has been preloaded to minimize the time in each indexing
        kitti_data.output_dict = self.output_dict
        calib, left_image, right_image, _, _ = kitti_data.read_data()
        calib.image_shape = left_image.shape

        transformed_left_image, transformed_right_image, P2, P3 = self.transform(
                left_image, right_image, deepcopy(calib.P2),deepcopy(calib.P3)
        )

        output_dict = {'calib': [P2, P3],
                       'image': [transformed_left_image, transformed_right_image],
                       'original_shape': calib.image_shape,
                       'original_P':calib.P2.copy()}
        return output_dict

    @staticmethod
    def collate_fn(batch):
        left_images = np.array([item["image"][0] for item in batch])#[batch, H, W, 3]
        left_images = left_images.transpose([0, 3, 1, 2])

        right_images = np.array([item["image"][1] for item in batch])#[batch, H, W, 3]
        right_images = right_images.transpose([0, 3, 1, 2])

        P2 = [item['calib'][0] for item in batch]
        P3 = [item['calib'][1] for item in batch]
        return torch.from_numpy(left_images).float(), torch.from_numpy(right_images).float(), P2, P3

