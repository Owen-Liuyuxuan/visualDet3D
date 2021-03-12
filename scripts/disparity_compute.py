from tqdm import tqdm
import numpy as np
import os
import pickle
import time
import cv2
from fire import Fire
from typing import List, Dict, Tuple
from copy import deepcopy
import skimage.measure
import torch

from _path_init import *
from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.networks.utils.utils import calc_iou, BBox3dProjector
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.data.kitti.kittidata import KittiData
from visualDet3D.data.kitti.utils import generate_dispariy_from_velo
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import cfg_from_file
def denorm(image:np.ndarray, rgb_mean:np.ndarray, rgb_std:np.ndarray)->np.ndarray:
    """
        Denormalize a image.
        Args:
            image: np.ndarray normalized [H, W, 3]
            rgb_mean: np.ndarray [3] among [0, 1] image
            rgb_std : np.ndarray [3] among [0, 1] image
        Returns:
            unnormalized image: np.ndarray (H, W, 3) [0-255] dtype=np.uint8
    """
    image = image * rgb_std + rgb_mean #
    image[image > 1] = 1
    image[image < 0] = 0
    image *= 255
    return np.array(image, dtype=np.uint8)

def process_train_val_file(cfg)-> Tuple[List[str], List[str]]:
    train_file = cfg.data.train_split_file
    val_file   = cfg.data.val_split_file

    with open(train_file) as f:
        train_lines = f.readlines()
        for i  in range(len(train_lines)):
            train_lines[i] = train_lines[i].strip()

    with open(val_file) as f:
        val_lines = f.readlines()
        for i  in range(len(val_lines)):
            val_lines[i] = val_lines[i].strip()

    return train_lines, val_lines

def compute_dispairity_for_split(cfg,
                                 index_names:List[str], 
                                 data_root_dir:str, 
                                 output_dict:Dict, 
                                 data_split:str='training', 
                                 time_display_inter:int=100, 
                                 use_point_cloud:bool=True):
    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    disp_dir = os.path.join(save_dir, 'disp')
    if not os.path.isdir(disp_dir):
        os.mkdir(disp_dir)

    if not use_point_cloud:
        stereo_matcher = cv2.StereoBM_create(192, 25)

    N = len(index_names)
    frames = [None] * N
    print("start reading {} data".format(data_split))
    timer = Timer()
    preprocess = build_augmentator(cfg.data.test_augmentation)

    for i, index_name in tqdm(enumerate(index_names)):

        # read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        calib, image, right_image, label, velo = data_frame.read_data()

        original_image = image.copy()
        baseline = (calib.P2[0, 3] - calib.P3[0, 3]) / calib.P2[0, 0]
        image, image_3, P2, P3 = preprocess(original_image, right_image.copy(), p2=deepcopy(calib.P2), p3=deepcopy(calib.P3))
        if use_point_cloud:
            ## gathering disparity with point cloud back projection
            disparity_left = generate_dispariy_from_velo(velo[:, 0:3], image.shape[0], image.shape[1], calib.Tr_velo_to_cam, calib.R0_rect, P2, baseline=baseline)
            disparity_right = generate_dispariy_from_velo(velo[:, 0:3], image.shape[0], image.shape[1], calib.Tr_velo_to_cam, calib.R0_rect, P3, baseline=baseline)

        else:
            ## gathering disparity with stereoBM from opencv
            left_image  = denorm(image, cfg.data.augmentation.rgb_mean, cfg.data.augmentation.rgb_std)
            right_image = denorm(image_3, cfg.data.augmentation.rgb_mean, cfg.data.augmentation.rgb_std)
            gray_image1 = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            gray_image2 = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

            disparity_left = stereo_matcher.compute(gray_image1, gray_image2)
            disparity_left[disparity_left < 0] = 0
            disparity_left = disparity_left.astype(np.uint16)

            disparity_right = stereo_matcher.compute(gray_image2[:, ::-1], gray_image1[:, ::-1])
            disparity_right[disparity_right < 0] = 0
            disparity_right= disparity_right.astype(np.uint16)

        
        disparity_left = skimage.measure.block_reduce(disparity_left, (4,4), np.max)
        file_name = os.path.join(disp_dir, "P2%06d.png" % i)
        cv2.imwrite(file_name, disparity_left)

        disparity_right = skimage.measure.block_reduce(disparity_right, (4,4), np.max)
        file_name = os.path.join(disp_dir, "P3%06d.png" % i)
        cv2.imwrite(file_name, disparity_left)



    print("{} split finished precomputing disparity".format(data_split))




def main(config:str="config/config.py",use_point_cloud:bool=False):
    """Main entry point for disparity precompute
    config_file(str): path to the config file.
    use_point_cloud(bool):  whether use OpenCV or point cloud to construct disparity ground truth.
    """
    cfg = cfg_from_file(config)
    torch.cuda.set_device(cfg.trainer.gpu)
    time_display_inter = 100 # define the inverval displaying time consumed in loop
    data_root_dir = cfg.path.data_path # the base directory of training dataset
    calib_path = os.path.join(data_root_dir, 'calib') 
    list_calib = os.listdir(calib_path)
    N = len(list_calib)
    # no need for image, could be modified for extended use
    output_dict = {
                "calib": True,
                "image": True,
                "image_3" : True,
                "label": False,
                "velodyne": use_point_cloud,
            }

    train_names, val_names = process_train_val_file(cfg)
    compute_dispairity_for_split(cfg, train_names, data_root_dir, output_dict, 'training', time_display_inter, use_point_cloud)

    print("Preprocessing finished")

if __name__ == '__main__':
    Fire(main)
