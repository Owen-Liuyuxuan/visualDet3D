"""
    Script for launching training process
"""
import os
import sys
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
from fire import Fire
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from _path_init import *
from visualDet3D.networks.utils.registry import DATASET_DICT
import visualDet3D.data.kitti.dataset
from visualDet3D.utils.utils import cfg_from_file

def compute_prior_map(w, h, K):

    x_range = np.arange(w, dtype=np.float32)
    y_range = np.arange(h, dtype=np.float32)
    _, yy_grid  = np.meshgrid(x_range, y_range) #[H, W]

    fy =  K[1:2, 1:2] #[1, 1]
    cy =  K[1:2, 2:3] #[1, 1]
    Ty =  0
    
    relative_elevation = 1.65
    depth = (fy * relative_elevation + Ty) / (yy_grid - cy + 1e-9) 
    
    prior = np.zeros_like(depth)
    mask = yy_grid > cy
    prior[mask] = np.log(depth[mask])
    
    prior[np.logical_not(mask)] = np.log(75)

    prior = np.clip(prior, 0, np.log(75))

    num = np.zeros_like(depth, dtype=np.long)
    num[mask] = 1000
    num[np.logical_not(mask)] = 10
    return prior * num, num

def precompute_depth_statistic(config:str="config/config.py", gpu=0):
    cfg = cfg_from_file(config)
    dataset_name = cfg.data.train_dataset
    dataset = DATASET_DICT[dataset_name](cfg, 'val') # being eval
    torch.cuda.set_device(gpu)

    save_dir = os.path.join(cfg.path.preprocessed_path, 'training')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    sum_data = torch.zeros([cfg.data.rgb_shape[0], cfg.data.rgb_shape[1]]).cuda() #[H,W]
    number_solid = torch.zeros_like(sum_data, dtype=torch.long).cuda()
    dataloader_train = DataLoader(dataset, num_workers=cfg.data.num_workers,
            batch_size=cfg.data.batch_size, collate_fn=dataset.collate_fn, shuffle=False, drop_last=False)

    N = len(dataset)
    for iter_num, data in tqdm(enumerate(dataloader_train)):
    # for i in tqdm(range(len(dataset))):
        #instance = dataset[i]
        sparse_depth = data[2].cuda()
        #sparse_depth = torch.tensor(instance['gt'], dtype=torch.float32).cuda() #[H, W] float, 0->useless
        log_depth    = torch.log(sparse_depth + 1e-9)

        for i in range(log_depth.shape[0]):
            mask = sparse_depth[i] > 0
            number_solid[mask] += 1
            sum_data[mask] += log_depth[i][mask]

    K = np.array(data[1][0]) #[3, 3]
    prior_map, weight_map = compute_prior_map(sum_data.shape[1], sum_data.shape[0], K)

    sum_file = os.path.join(save_dir,'log_depth_sum.npy')
    np.save(sum_file, sum_data.cpu().numpy() + prior_map)
    num_file = os.path.join(save_dir,'log_depth_solid.npy')
    np.save(num_file, number_solid.cpu().numpy()+ weight_map)

if __name__ == "__main__":
    Fire(precompute_depth_statistic)
