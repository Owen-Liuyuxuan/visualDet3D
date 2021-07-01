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
from visualDet3D.data.kitti.dataset import KittiMonoDataset
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.utils.timer import profile
from visualDet3D.networks.utils.rtm3d_utils import gen_hm_radius, project_to_image, gaussian_radius
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
class KittiRTM3DDataset(KittiMonoDataset):
    def __init__(self, cfg, split='training'):
        super(KittiRTM3DDataset, self).__init__(cfg, split)
        self.num_classes = len(self.obj_types)
        self.num_vertexes = 9
        self.max_objects = 32
        self.projector.register_buffer('corner_matrix', torch.tensor(
            [[-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [ 1,  1,  1],
            [ 1, -1,  1],
            [-1, -1,  1],
            [-1,  1,  1],
            [-1,  1, -1],
            [ 0,  0,  0]]
        ).float()  )# 9, 3

    def _build_target(self, image:np.ndarray, P2:np.ndarray, transformed_label:List[KittiObj], scale=4)-> dict:
        """Encode Targets for RTM3D

        Args:
            image (np.ndarray): augmented image [H, W, 3]
            P2 (np.ndarray): Calibration matrix [3, 4]
            transformed_label (List[KittiObj]): A list of kitti objects.
            scale (int, optional): Downsampling scale. Defaults to 4.

        Returns:
            dict: label dicts
        """        
        num_objects = len(transformed_label)
        hm_h, hm_w = image.shape[0] // scale, image.shape[1] // scale

        # setup empty targets
        hm_main_center = np.zeros((self.num_classes, hm_h, hm_w), dtype=np.float32)
        hm_ver = np.zeros((self.num_vertexes, hm_h, hm_w), dtype=np.float32)

        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        indices_center = np.zeros((self.max_objects), dtype=np.int64)
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8)
        location = np.zeros((self.max_objects, 3), dtype=np.float32)
        orientation = np.zeros((self.max_objects, 1), dtype=np.float32)
        rotbin = np.zeros((self.max_objects, 2), dtype=np.int64)
        rotres = np.zeros((self.max_objects, 2), dtype=np.float32)
        ver_coor = np.zeros((self.max_objects, self.num_vertexes * 2), dtype=np.float32)
        ver_coor_mask = np.zeros((self.max_objects, self.num_vertexes * 2), dtype=np.uint8)
        ver_offset = np.zeros((self.max_objects * self.num_vertexes, 2), dtype=np.float32)
        ver_offset_mask = np.zeros((self.max_objects * self.num_vertexes), dtype=np.uint8)
        indices_vertexes = np.zeros((self.max_objects * self.num_vertexes), dtype=np.int64)

        dimension = np.zeros((self.max_objects, 3), dtype=np.float32)

        rots = np.zeros((self.max_objects, 2), dtype=np.float32) #[sin, cos]

        depth = np.zeros((self.max_objects, 1), dtype=np.float32)
        whs = np.zeros((self.max_objects, 2), dtype=np.float32)

        # compute vertexes
        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        for obj in transformed_label:
            obj.alpha = theta2alpha_3d(obj.ry, obj.x, obj.z, P2)
        bbox3d_origin = torch.tensor([[obj.x, obj.y - 0.5 * obj.h, obj.z, obj.w, obj.h, obj.l, obj.alpha] for obj in transformed_label], dtype=torch.float32).reshape(-1, 7)
        abs_corner, homo_corner, theta = self.projector.forward(bbox3d_origin, torch.tensor(P2, dtype=torch.float32))

        # # For debuging and visualization, testing the correctness of bbox3d->bbox2d
        # a = plt.figure(figsize=(16,9))
        # plt.subplot(3, 1, 1)
        # image2 = np.array(np.clip(image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1) * 255, dtype=np.uint8)
        # max_xy, _= homo_corner[:, :, 0:2].max(dim = 1)  # [N,2]
        # min_xy, _= homo_corner[:, :, 0:2].min(dim = 1)  # [N,2]

        # result = torch.cat([min_xy, max_xy], dim=-1) #[:, 4]

        # bbox2d = result.cpu().numpy()
        # for i in range(len(transformed_label)):
        #     image2 = cv2.rectangle(image2, tuple(bbox2d[i, 0:2].astype(int)), tuple(bbox2d[i, 2:4].astype(int)), (0, 255, 0) , 3)
        #     draw_3D_box(image2, homo_corner[i].cpu().numpy().T)
        # plt.imshow(image2)
        # plt.show()


        for k in range(num_objects):
            obj = transformed_label[k]
            cls_id = self.obj_types.index(obj.type)
            bbox = np.array([obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b])
            orientation[k] = obj.ry
            dim  = np.array([obj.w, obj.h, obj.l])
            ry   = obj.ry
            alpha= obj.alpha

            if np.sin(alpha) < 0.5: #alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                rotbin[k, 0] = 1
                rotres[k, 0] = alpha - (-0.5 * np.pi)
            if np.sin(alpha) > -0.5: # alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                rotbin[k, 1] = 1
                rotres[k, 1] = alpha - (0.5 * np.pi)

            bbox = bbox / scale  # on the heatmap
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, image.shape[1] // scale)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, image.shape[0] // scale)
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if bbox_h > 0 and bbox_w > 0:
                sigma = 1.  # Just dummy
                radius = 1  # Just dummy

                location[k] = bbox3d_origin[k, 0:3].float().cpu().numpy()

                radius = gaussian_radius((np.ceil(bbox_h), np.ceil(bbox_w)))
                radius = max(0, int(radius))
                # Generate heatmaps for 8 vertexes
                vertexes_2d = homo_corner[k, :, 0:2].numpy()

                vertexes_2d = vertexes_2d / scale  # on the heatmap

                center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)
                
                if not (0 <= center_int[0] < hm_w and 0 <= center_int[1] < hm_h):
                    continue

                # Generate heatmaps for main center
                gen_hm_radius(hm_main_center[cls_id], center, radius)
                # Index of the center
                indices_center[k] = center_int[1] * hm_w + center_int[0]

                for ver_idx, ver in enumerate(vertexes_2d):
                    ver_int = ver.astype(np.int32)
                    
                    # targets for vertexes coordinates
                    ver_coor[k, ver_idx * 2: (ver_idx + 1) * 2] = ver - center_int  # Don't take the absolute values
                    ver_coor_mask[k, ver_idx * 2: (ver_idx + 1) * 2] = 1
                    
                    if (0 <= ver_int[0] < hm_w) and (0 <= ver_int[1] < hm_h):
                        gen_hm_radius(hm_ver[ver_idx], ver_int, radius)
                        
                        # targets for vertexes offset
                        ver_offset[k * self.num_vertexes + ver_idx] = ver - ver_int
                        ver_offset_mask[k * self.num_vertexes + ver_idx] = 1
                        # Indices of vertexes
                        indices_vertexes[k * self.num_vertexes + ver_idx] = ver_int[1] * hm_w + ver_int[0]

                # targets for center offset
                cen_offset[k] = center - center_int

                # targets for dimension
                dimension[k] = dim

                # targets for orientation
                rots[k, 0] = np.sin(alpha)
                rots[k, 1] = np.cos(alpha)

                # targets for depth
                depth[k] = obj.z

                # targets for 2d bbox
                whs[k, 0] = bbox_w
                whs[k, 1] = bbox_h

                # Generate masks
                obj_mask[k] = 1
        # Follow official names
        targets = {
            'hm': hm_main_center,
            'hm_hp': hm_ver,
            'hps': ver_coor,
            'reg': cen_offset,
            'hp_offset': ver_offset,
            'dim': dimension, #whl
            'rots': rots, # sin cos alpha
            'rotbin': rotbin,
            'rotres': rotres,
            'dep': depth,
            'ind': indices_center,
            'hp_ind': indices_vertexes,
            'reg_mask': obj_mask,
            'hps_mask': ver_coor_mask,
            'hp_mask': ver_offset_mask,
            'wh': whs,
            'location': location,
            'ori': orientation
        }

        return targets

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
        targets = self._build_target(transformed_image, transformed_P2, transformed_label)
        
        output_dict = {'calib': transformed_P2,
                       'image': transformed_image,
                       'label': targets, 
                       'original_shape':image.shape,
                       'original_P':calib.P2.copy()}
        
        return output_dict
        
    
    def __len__(self):
        return len(self.imdb)

    @staticmethod
    def collate_fn(batch):
        rgb_images = np.array([item["image"] for item in batch])#[batch, H, W, 3]
        rgb_images = rgb_images.transpose([0, 3, 1, 2])

        calib = [item["calib"] for item in batch]

        # gather labels
        label = {}
        for key in batch[0]['label']:
            label[key] = torch.from_numpy(
                np.stack(
                    [
                        item['label'][key] for item in batch
                    ], axis=0
                )
            )
        return torch.from_numpy(rgb_images).float(), torch.tensor(calib).float(), label

@DATASET_DICT.register_module
class KittiMonoFlexDataset(KittiRTM3DDataset):
    def __init__(self, cfg, split='training'):
        super(KittiRTM3DDataset, self).__init__(cfg, split)
        self.num_classes = len(self.obj_types)
        self.num_vertexes = 10
        self.max_objects = 32
        self.projector.register_buffer('corner_matrix', torch.tensor(
            [[-1, -1, -1], #0
            [ 1, -1, -1],  #1
            [ 1,  1, -1],  #2
            [ 1,  1,  1],  #3
            [ 1, -1,  1],  #4
            [-1, -1,  1],  #5
            [-1,  1,  1],  #6
            [-1,  1, -1],  #7
            [ 0,  1,  0],  #8
            [ 0, -1,  0],  #9
            [ 0,  0,  0]]  #10
        ).float()  )# 9, 3

    def _get_edge_utils(self, image_size:Tuple[int, int], down_ratio=4):
        img_w, img_h = image_size

        x_min, y_min = 0, 0
        x_max, y_max = image_size[0] // down_ratio, image_size[1] // down_ratio

        step=1
        # boundary idxs
        edge_indices = []

        # left
        y = np.arange(y_min, y_max, step)
        x = np.ones(len(y)) * x_min

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = np.arange(x_min, x_max, step)
        y = np.ones(len(x)) * y_max

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices.append(edge_indices_edge)

        # right
        y = np.arange(y_max, y_min, -step)
        x = np.ones(len(y)) * x_max

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices.append(edge_indices_edge)

        # top  
        x = np.arange(x_max, x_min - 1, -step)
        y = np.ones(len(x)) * y_min

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices.append(edge_indices_edge)

        # concatenate
        edge_indices = np.concatenate([index.astype(np.long) for index in edge_indices], axis=0)
        edge_indices = np.unique(edge_indices, axis=0)

        return edge_indices


    def _build_target(self, image:np.ndarray, P2:np.ndarray, transformed_label:List[KittiObj], scale=4)-> dict:
        """Encode Targets for MonoFlex

        Args:
            image (np.ndarray): augmented image [H, W, 3]
            P2 (np.ndarray): Calibration matrix [3, 4]
            transformed_label (List[KittiObj]): A list of kitti objects.
            scale (int, optional): Downsampling scale. Defaults to 4.

        Returns:
            dict: label dicts
        """        
        num_objects = len(transformed_label)
        hm_h, hm_w = image.shape[0] // scale, image.shape[1] // scale

        # setup empty targets
        hm_main_center = np.zeros((self.num_classes, hm_h, hm_w), dtype=np.float32)
        hm_ver = np.zeros((self.num_vertexes, hm_h, hm_w), dtype=np.float32)

        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        indices_center = np.zeros((self.max_objects), dtype=np.int64)
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8)
        bboxes2d = np.zeros((self.max_objects, 4), dtype=np.float32)
        fcos_bbox2d_target = np.zeros((self.max_objects, 4), dtype=np.float32)
        location = np.zeros((self.max_objects, 3), dtype=np.float32)
        orientation = np.zeros((self.max_objects, 1), dtype=np.float32)
        rotbin = np.zeros((self.max_objects, 2), dtype=np.int64)
        rotres = np.zeros((self.max_objects, 2), dtype=np.float32)
        ver_coor = np.zeros((self.max_objects, self.num_vertexes * 2), dtype=np.float32)
        ver_coor_mask = np.zeros((self.max_objects, self.num_vertexes * 2), dtype=np.uint8)
        ver_offset = np.zeros((self.max_objects * self.num_vertexes, 2), dtype=np.float32)
        ver_offset_mask = np.zeros((self.max_objects * self.num_vertexes), dtype=np.uint8)
        indices_vertexes = np.zeros((self.max_objects * self.num_vertexes), dtype=np.int64)
        keypoints_depth_mask = np.zeros((self.max_objects, 3), dtype=np.float32)

        dimension = np.zeros((self.max_objects, 3), dtype=np.float32)

        rots = np.zeros((self.max_objects, 2), dtype=np.float32) #[sin, cos]

        depth = np.zeros((self.max_objects, 1), dtype=np.float32)
        whs = np.zeros((self.max_objects, 2), dtype=np.float32)

        # compute vertexes
        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        for obj in transformed_label:
            obj.alpha = theta2alpha_3d(obj.ry, obj.x, obj.z, P2)
        bbox3d_origin = torch.tensor([[obj.x, obj.y - 0.5 * obj.h, obj.z, obj.w, obj.h, obj.l, obj.alpha] for obj in transformed_label], dtype=torch.float32).reshape(-1, 7)
        abs_corner, homo_corner, theta = self.projector.forward(bbox3d_origin, torch.tensor(P2, dtype=torch.float32))

        ## Different from RTM3D: edge fusion
        edge_indices = self._get_edge_utils((image.shape[0], image.shape[1]), down_ratio=4)

        for k in range(num_objects):
            obj = transformed_label[k]
            cls_id = self.obj_types.index(obj.type)
            bbox = np.array([obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b])
            orientation[k] = obj.ry
            dim  = np.array([obj.w, obj.h, obj.l])
            ry   = obj.ry
            alpha= obj.alpha

            if np.sin(alpha) < 0.5: #alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                rotbin[k, 0] = 1
                rotres[k, 0] = alpha - (-0.5 * np.pi)
            if np.sin(alpha) > -0.5: # alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                rotbin[k, 1] = 1
                rotres[k, 1] = alpha - (0.5 * np.pi)

            bbox = bbox / scale  # on the heatmap
            bboxes2d[k] = bbox
            
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, image.shape[1] // scale)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, image.shape[0] // scale)
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if bbox_h > 0 and bbox_w > 0:
                sigma = 1.  # Just dummy
                radius = 1  # Just dummy

                location[k] = bbox3d_origin[k, 0:3].float().cpu().numpy()

                radius = gaussian_radius((np.ceil(bbox_h), np.ceil(bbox_w)))
                radius = max(0, int(radius))
                ## Different from RTM3D:
                # Generate heatmaps for 10 vertexes
                vertexes_2d = homo_corner[k, 0:10, 0:2].numpy()

                vertexes_2d = vertexes_2d / scale  # on the heatmap

                # keypoints mask: keypoint must be inside the image and in front of the camera
                keypoints_x_visible = (vertexes_2d[:, 0] >= 0) & (vertexes_2d[:, 0] <= hm_w)
                keypoints_y_visible = (vertexes_2d[:, 1] >= 0) & (vertexes_2d[:, 1] <= hm_h)
                keypoints_z_visible = (abs_corner[k, 0:10, 2].numpy() > 0)
                keypoints_visible   = keypoints_x_visible & keypoints_y_visible & keypoints_z_visible
                keypoints_visible = np.append(
                    np.tile(keypoints_visible[:4] | keypoints_visible[4:8], 2), np.tile(keypoints_visible[8] | keypoints_visible[9], 2)
                ) # "modified keypoint visible from monoflex"
                keypoints_depth_valid = np.stack(
                    (keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(), keypoints_visible[[1, 3, 5, 7]].all())
                ).astype(np.float32)
                keypoints_visible = keypoints_visible.astype(np.float32)

                ## MonoFlex use the projected 3D as the center
                #center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                center = homo_corner[k, 10, 0:2].numpy() / scale
                center_int = center.astype(np.int32)
                
                if not (0 <= center_int[0] < hm_w and 0 <= center_int[1] < hm_h):
                    continue

                # Generate heatmaps for main center
                gen_hm_radius(hm_main_center[cls_id], center, radius)
                # Index of the center
                indices_center[k] = center_int[1] * hm_w + center_int[0]

                for ver_idx, ver in enumerate(vertexes_2d):
                    ver_int = ver.astype(np.int32)
                    
                    # targets for vertexes coordinates
                    ver_coor[k, ver_idx * 2: (ver_idx + 1) * 2] = ver - center_int  # Don't take the absolute values
                    ver_coor_mask[k, ver_idx * 2: (ver_idx + 1) * 2] = 1
                    
                    if (0 <= ver_int[0] < hm_w) and (0 <= ver_int[1] < hm_h):
                        gen_hm_radius(hm_ver[ver_idx], ver_int, radius)
                        
                        # targets for vertexes offset
                        ver_offset[k * self.num_vertexes + ver_idx] = ver - ver_int
                        ver_offset_mask[k * self.num_vertexes + ver_idx] = 1
                        # Indices of vertexes
                        indices_vertexes[k * self.num_vertexes + ver_idx] = ver_int[1] * hm_w + ver_int[0]

                # targets for center offset
                cen_offset[k] = center - center_int

                ## targets for fcos 2d
                fcos_bbox2d_target[k] = np.array(
                    [center_int[0] - bbox[0], center_int[1] - bbox[1], bbox[2] - center_int[0], bbox[3] - center_int[1]]
                )
                # targets for dimension
                dimension[k] = dim

                # targets for orientation
                rots[k, 0] = np.sin(alpha)
                rots[k, 1] = np.cos(alpha)

                # targets for depth
                depth[k] = obj.z

                # targets for 2d bbox
                whs[k, 0] = bbox_w
                whs[k, 1] = bbox_h

                # Generate masks
                obj_mask[k] = 1
                keypoints_depth_mask[k] = keypoints_depth_valid

        # Follow official names
        targets = {
            'hm': hm_main_center,
            'hm_hp': hm_ver,
            'hps': ver_coor,
            'reg': cen_offset,
            'hp_offset': ver_offset,
            'dim': dimension, #whl
            'rots': rots, # sin cos alpha
            'rotbin': rotbin,
            'rotres': rotres,
            'dep': depth,
            'ind': indices_center,
            'hp_ind': indices_vertexes,
            'reg_mask': obj_mask,
            'hps_mask': ver_coor_mask,
            'hp_mask': ver_offset_mask,
            'kp_detph_mask': keypoints_depth_mask,
            'wh': whs,
            'bboxes2d': bboxes2d,
            'bboxes2d_target': fcos_bbox2d_target,
            'location': location,
            'ori': orientation,
            'edge_indices': edge_indices
        }

        return targets
