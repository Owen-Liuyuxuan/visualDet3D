"""
    This script contains function snippets for different training settings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from easydict import EasyDict
from visualDet3D.utils.utils import LossLogger, compound_annotation
from visualDet3D.networks.utils.registry import PIPELINE_DICT
from typing import Tuple, List

@PIPELINE_DICT.register_module
@torch.no_grad()
def test_mono_detection(data, module:nn.Module,
                     writer:SummaryWriter, 
                     loss_logger:LossLogger=None, 
                     global_step:int=None, 
                     cfg:EasyDict=None)-> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    image, P2 = data[0], data[1]

    scores, bbox, obj_index = module(
        [image.cuda().float().contiguous(), torch.tensor(P2).cuda().float()])
    obj_types = [cfg.obj_types[i.item()] for i in obj_index]

    return scores, bbox, obj_types

@PIPELINE_DICT.register_module
@torch.no_grad()
def test_stereo_detection(data, module:nn.Module,
                     writer:SummaryWriter, 
                     loss_logger:LossLogger=None, 
                     global_step:int=None, 
                     cfg:EasyDict=None) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    left_images, right_images, P2, P3 = data[0], data[1], data[2], data[3]

    scores, bbox, obj_index = module([left_images.cuda().float().contiguous(), right_images.cuda().float().contiguous(), torch.tensor(P2).cuda().float(), torch.tensor(P3).cuda().float()])
    obj_types = [cfg.obj_types[i.item()] for i in obj_index]

    return scores, bbox, obj_types

@PIPELINE_DICT.register_module
@torch.no_grad()
def test_mono_detection_diggingmono3d(data, module:nn.Module,
                     writer:SummaryWriter, 
                     loss_logger:LossLogger=None, 
                     global_step:int=None, 
                     cfg:EasyDict=EasyDict())-> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    image, P2 = data[0], data[1]

    scores, bbox, obj_index = module(
        [image.cuda().float().contiguous(), torch.tensor(P2).cuda().float()])
    
    if len(scores) == 0:
        return scores, bbox, obj_index
    
    expanded_scores = []
    expanded_bboxes = []
    expanded_obj_index = []
    
    for i in range(len(scores)):
        expanded_scores.append(scores[i])
        expanded_bboxes.append(bbox[i])
        expanded_obj_index.append(obj_index[i])
        depth = bbox[i, 6]
        if depth < 10:
            # no augmentation for these measurements
            continue
        
        lamda = 80
        for depth_shift in [-2, -1, -0.5, 0.5, 1, 2]:
            new_depth = depth + depth_shift
            new_bbox = bbox[i].clone()
            new_bbox[6] = new_depth
            sigma = torch.exp(depth / lamda)
            t_s = torch.exp( - depth_shift**2 / sigma**2)
            new_score = scores[i] * t_s
            expanded_scores.append(new_score)
            expanded_bboxes.append(new_bbox)
            expanded_obj_index.append(obj_index[i])
        
    expanded_scores = torch.stack(expanded_scores)
    expanded_bboxes = torch.stack(expanded_bboxes, dim=0)
        
    
    obj_types = [cfg.obj_types[i.item()] for i in expanded_obj_index]

    return expanded_scores, expanded_bboxes, obj_types
