from copy import deepcopy
import torch
import cv2
import time
import pickle
import os
import numpy as np
import ctypes

from _path_init import *
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import cfg_from_file
from visualDet3D.data.kitti.kittidata import KittiData



def read_one_split(cfg, index_names, data_root_dir, output_dict, data_split='training', time_display_inter=100):
    N = len(index_names)
    frames = [None] * N
    print("start reading {} data".format(data_split))
    timer = Timer()

    for i, index_name in enumerate(index_names):

        # read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        calib, _, _, _ = data_frame.read_data()

        # store the list of kittiObjet and kittiCalib
        data_frame.calib = calib

        frames[i] = data_frame

        if (i+1) % time_display_inter == 0:
            avg_time = timer.compute_avg_time(i+1)
            eta = timer.compute_eta(i+1, N)
            print("{} iter:{}/{}, avg-time:{}, eta:{}".format(
                data_split, i+1, N, avg_time, eta), end='\r')

    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    pkl_file = os.path.join(save_dir, 'imdb.pkl')
    pickle.dump(frames, open(pkl_file, 'wb'))
    print("{} split finished precomputing".format(data_split))


def main(config="config/config.py"):
    cfg = cfg_from_file(config)
    torch.cuda.set_device(cfg.trainer.gpu)
    time_display_inter = 100  # define the inverval displaying time consumed in loop
    data_root_dir = cfg.path.test_path  # the base directory of training dataset
    calib_path = os.path.join(data_root_dir, 'calib')
    list_calib = os.listdir(calib_path)
    N = len(list_calib)
    # no need for image, could be modified for extended use
    output_dict = {
        "calib": True,
        "image": False,
        "label": False,
        "velodyne": False,
    }

    num_test_file = N
    test_names = ["%06d" % i for i in range(num_test_file)]
    read_one_split(cfg, test_names, data_root_dir, output_dict,
                   'test', time_display_inter)

    print("Preprocessing finished")


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
