from __future__ import print_function, division
from multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from visualDet3D.data.kitti.utils import read_image, read_pc_from_bin, read_depth
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.networks.utils.registry import DATASET_DICT
from PIL import Image
import os
import pickle
import numpy as np
from copy import deepcopy
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
ros_py_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
import skimage.measure
if sys.version_info > (3, 0) and ros_py_path in sys.path:
    #Python 3, compatible with a naive ros environment
    sys.path.remove(ros_py_path)
    import cv2
    sys.path.append(ros_py_path)
else:
    #Python 2
    import cv2

def read_K_from_depth_prediction(file):
    with open(file, 'r') as f:
        line = f.readlines()[0]
        data = line.split(" ")
        K    = np.array([float(data[i]) for i in range(len(data[0:9]))])
        return np.reshape(K, (3, 3))

def read_P23_from_sequence(file):
    """ read P2 and P3 from a sequence file calib_cam_to_cam.txt
    """
    P2 = None
    P3 = None
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith("P_rect_02"):
                data = line.split(" ")
                P2 = np.array([float(x) for x in data[1:13]])
                P2 = np.reshape(P2, [3, 4])
            if line.startswith("P_rect_03"):
                data = line.split(" ")
                P3 = np.array([float(x) for x in data[1:13]])
                P3 = np.reshape(P3, [3, 4])
    assert P2 is not None, f"can not find P2 in file {file}"
    assert P3 is not None, f"can not find P3 in file {file}"
    return P2, P3

def read_T_from_sequence(file):
    """ read T from a sequence file calib_velo_to_cam.txt
    """
    R = None
    T = None
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith("R:"):
                data = line.split(" ")
                R = np.array([float(x) for x in data[1:10]])
                R = np.reshape(R, [3, 3])
            if line.startswith("T:"):
                data = line.split(" ")
                T = np.array([float(x) for x in data[1:4]])
                T = np.reshape(T, [3, 1]) 
    assert R is not None, "can not find R in file {}".format(file)
    assert T is not None, "can not find T in file {}".format(file)

    T_velo2cam = np.eye(4)
    T_velo2cam[0:3, 0:3] = R
    T_velo2cam[0:3, 3:4] = T
    return T_velo2cam

@DATASET_DICT.register_module
class KittiDepthMonoDataset(torch.utils.data.Dataset):
    """Some Information about KittiDataset"""
    def __init__(self, cfg, split='training'):
        super(KittiDepthMonoDataset, self).__init__()
        raw_path    = cfg.path.raw_path
        depth_paths  = cfg.path.depth_path if isinstance(cfg.path.depth_path, list) else [cfg.path.depth_path]
        aug_cfg     = cfg.data.augmentation
        manager = Manager() # multithread manage wrapping for list objects
        self.is_eval = not split == 'training'
        self.size = aug_cfg.cropSize #[352, 1216]
        raw_sequences = {}
        for date_time in os.listdir(raw_path):
            sequences = os.listdir(os.path.join(raw_path, date_time))
            sequences = [sequence for sequence in sequences if not sequence.endswith(".txt")]
            P2, P3 = read_P23_from_sequence(os.path.join(raw_path, date_time, "calib_cam_to_cam.txt"))
            T      = read_T_from_sequence  (os.path.join(raw_path, date_time, "calib_velo_to_cam.txt"))
            for sequence in sequences:
                raw_sequences[sequence] = dict(P2=P2, P3=P3, T_velo2cam=T, date_time=date_time)
        self.imdb = []
        print("Start counting images in depth prediction dataset.")
        for depth_path in depth_paths:
            for sequence in tqdm(os.listdir(depth_path)):
                sequence_gt_path = os.path.join(depth_path, sequence, 'proj_depth', 'groundtruth')
                P2 = raw_sequences[sequence]['P2']
                P3 = raw_sequences[sequence]['P3']
                T  = raw_sequences[sequence]['T_velo2cam']
                left_gt_dir = os.path.join(sequence_gt_path, 'image_02')
                right_gt_dir = os.path.join(sequence_gt_path, 'image_03')
                gt_names = set(os.listdir(left_gt_dir))

                left_image_dir = os.path.join(raw_path, raw_sequences[sequence]['date_time'], sequence, 'image_02', 'data')
                right_image_dir = os.path.join(raw_path, raw_sequences[sequence]['date_time'], sequence, 'image_03', 'data')
                point_cloud_dir = os.path.join(raw_path, raw_sequences[sequence]['date_time'], sequence, 'velodyne_points', 'data')
                image_names = set(os.listdir(left_image_dir))

                intersection = gt_names.intersection(image_names) # names in both
                instances = [
                    dict(
                        image_2_file = os.path.join(left_image_dir, name),
                        image_3_file = os.path.join(right_image_dir, name),
                        gt_2_file    = os.path.join(left_gt_dir, name),
                        gt_3_file    = os.path.join(right_gt_dir, name),
                        P2           = P2.copy(),
                        P3           = P3.copy(),
                        # T_velo2cam   = T.copy(),
                        # point_cloud_file = os.path.join(point_cloud_dir, name.replace('.png', '.bin'))
                    ) for name in sorted(intersection)
                ]
                self.imdb += instances

        self.imdb = manager.list(self.imdb)
        if not self.is_eval:
            self.transform = build_augmentator(cfg.data.train_augmentation)
        else:
            self.transform = build_augmentator(cfg.data.test_augmentation)

    def __getitem__(self, index):
        obj = self.imdb[index]
        # point_cloud = read_pc_from_bin(obj['point_cloud_file'])[..., 0:3]  #[-1, 4]
        # T_velo2cam  = obj['T_velo2cam']
        if self.is_eval or np.random.rand() < 0.5: # Randomly select left/right image
            image = read_image(obj['image_2_file'])
            gt    = read_depth(obj['gt_2_file'])
            P     = obj['P2']
        else:
            image = read_image(obj['image_3_file'])
            gt    = read_depth(obj['gt_3_file'])
            P     = obj['P3']
        
        transformed_image, P_new, gt = self.transform(image, p2=P.copy(), image_gt=gt)
        output_dict = {'K': P_new[0:3, 0:3].copy(),
                       'image': transformed_image,
                       'gt': gt,
                       'original_shape': image.shape}
        return output_dict

    def __len__(self):
        return len(self.imdb)

    @staticmethod
    def collate_fn(batch):
        rgb_images = np.array([item["image"] for item in batch])#[batch, H, W, 3]
        rgb_images = rgb_images.transpose([0, 3, 1, 2])

        Ks = [item["K"] for item in batch]
        gts  = np.stack([item["gt"] for item in batch], axis=0) #[B, H, W]
        return torch.from_numpy(rgb_images).float(), Ks, torch.from_numpy(gts).float()

@DATASET_DICT.register_module
class KittiDepthMonoValTestDataset(torch.utils.data.Dataset):
    """Some Information about KittiDataset"""
    def __init__(self, cfg, split='validation'):
        super(KittiDepthMonoValTestDataset, self).__init__()
        base_path = cfg.path[split + "_path"]
        self.transform = build_augmentator(cfg.data.test_augmentation)

        self.imdb = []
        image_dir = os.path.join(base_path, "image")
        intrinsic_dir = os.path.join(base_path, "intrinsics")

        image_list = os.listdir(image_dir)
        image_list.sort()

        K_list = os.listdir(intrinsic_dir)
        K_list.sort()
        self.imdb = [
            dict(
                image_file = os.path.join(image_dir, image_list[i]),
                K          = read_K_from_depth_prediction(os.path.join(intrinsic_dir, K_list[i]))
            ) for i in range(len(image_list))
        ]

    def __getitem__(self, index):
        obj = self.imdb[index]
        image = read_image(obj['image_file'])
        K     = obj['K'].copy()
        
        transformed_image = self.transform(image)[0] # shape should not change since input output should all be 352 * 1216
        output_dict = {'K': K,
                       'image': transformed_image,
                       'original_shape': image.shape}
        return output_dict

    def __len__(self):
        return len(self.imdb)

    @staticmethod
    def collate_fn(batch):
        rgb_images = np.array([item["image"] for item in batch]) #[batch, H, W, 3]
        rgb_images = rgb_images.transpose([0, 3, 1, 2])

        Ks = [item["K"] for item in batch]
        return torch.from_numpy(rgb_images).float(), Ks
