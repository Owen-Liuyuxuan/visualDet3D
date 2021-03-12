"""
This file contains all PyTorch data augmentation functions.

Every transform should have a __call__ function which takes in (self, image, imobj)
where imobj is an arbitary dict containing relevant information to the image.

In many cases the imobj can be None, which enables the same augmentations to be used
during testing as they are in training.

Optionally, most transforms should have an __init__ function as well, if needed.
"""

import numpy as np
from numpy import random
import cv2
import math
import os
import sys
from easydict import EasyDict
from typing import List
from matplotlib import pyplot as plt
from visualDet3D.networks.utils.utils import BBox3dProjector
from visualDet3D.utils.utils import draw_3D_box, theta2alpha_3d
from visualDet3D.networks.utils.registry import AUGMENTATION_DICT
from visualDet3D.data.kitti.kittidata import KittiObj
import torch
from .augmentation_builder import Compose, build_single_augmentator

@AUGMENTATION_DICT.register_module
class ConvertToFloat(object):
    """
    Converts image data type to float.
    """
    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        return left_image.astype(np.float32), right_image if right_image is None else right_image.astype(np.float32), p2, p3, labels, image_gt, lidar


@AUGMENTATION_DICT.register_module
class Normalize(object):
    """
    Normalize the image
    """
    def __init__(self, mean, stds):
        self.mean = np.array(mean, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        left_image = left_image.astype(np.float32)
        left_image /= 255.0
        left_image -= np.tile(self.mean, int(left_image.shape[2]/self.mean.shape[0]))
        left_image /= np.tile(self.stds, int(left_image.shape[2]/self.stds.shape[0]))
        left_image.astype(np.float32)
        if right_image is not None:
            right_image = right_image.astype(np.float32)
            right_image /= 255.0
            right_image -= np.tile(self.mean, int(right_image.shape[2]/self.mean.shape[0]))
            right_image /= np.tile(self.stds, int(right_image.shape[2]/self.stds.shape[0]))
            right_image = right_image.astype(np.float32)
        return left_image, right_image, p2, p3, labels, image_gt, lidar


@AUGMENTATION_DICT.register_module
class Resize(object):
    """
    Resize the image according to the target size height and the image height.
    If the image needs to be cropped after the resize, we crop it to self.size,
    otherwise we pad it with zeros along the right edge

    If the object has ground truths we also scale the (known) box coordinates.
    """
    def __init__(self, size, preserve_aspect_ratio=True):
        self.size = size
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):

        if self.preserve_aspect_ratio:
            scale_factor = self.size[0] / left_image.shape[0]

            h = np.round(left_image.shape[0] * scale_factor).astype(int)
            w = np.round(left_image.shape[1] * scale_factor).astype(int)
            
            scale_factor_yx = (scale_factor, scale_factor)
        else:
            scale_factor_yx = (self.size[0] / left_image.shape[0], self.size[1] / left_image.shape[1])

            h = self.size[0]
            w = self.size[1]

        # resize
        left_image = cv2.resize(left_image, (w, h))
        if right_image is not None:
            right_image = cv2.resize(right_image, (w, h))
        if image_gt is not None:
            image_gt = cv2.resize(image_gt, (w, h), cv2.INTER_NEAREST)

        if len(self.size) > 1:

            # crop in
            if left_image.shape[1] > self.size[1]:
                left_image = left_image[:, 0:self.size[1], :]
                if right_image is not None:
                    right_image = right_image[:, 0:self.size[1], :]
                if image_gt is not None:
                    image_gt = image_gt[:, 0:self.size[1]]

            # pad out
            elif left_image.shape[1] < self.size[1]:
                padW = self.size[1] - left_image.shape[1]
                left_image  = np.pad(left_image,  [(0, 0), (0, padW), (0, 0)], 'constant')
                if right_image is not None:
                    right_image = np.pad(right_image, [(0, 0), (0, padW), (0, 0)], 'constant')
                if image_gt is not None:
                    if len(image_gt.shape) == 2:
                        image_gt = np.pad(image_gt, [(0, 0), (0, padW)], 'constant')
                    else:
                        image_gt = np.pad(image_gt, [(0, 0), (0, padW), (0, 0)], 'constant')

        if p2 is not None:
            p2[0, :]   = p2[0, :] * scale_factor_yx[1]
            p2[1, :]   = p2[1, :] * scale_factor_yx[0]
        
        if p3 is not None:
            p3[0, :]   = p3[0, :] * scale_factor_yx[1]
            p3[1, :]   = p3[1, :] * scale_factor_yx[0]
        
        if labels:
            if isinstance(labels, list):
                for obj in labels:
                    obj.bbox_l *= scale_factor_yx[1]
                    obj.bbox_r *= scale_factor_yx[1]
                    obj.bbox_t *= scale_factor_yx[0]
                    obj.bbox_b *= scale_factor_yx[0]
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class ResizeToFx(object):
    """
    Resize the image so that the Fx is aligned to a preset value

    If the object has ground truths we also scale the (known) box coordinates.
    """
    def __init__(self, Fx=721.5337, Fy=None):
        self.Fx = Fx
        self.Fy = Fy if Fy is not None else Fx

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if p2 is None:
            print("P2 is None in ResizeToFx, will return the original input")
            return left_image, right_image, p2, p3, labels, image_gt, lidar
        
        h0 = left_image.shape[0]
        w0 = left_image.shape[1]
        fx0 = p2[0, 0]
        fy0 = p2[1, 1]

        h1 = int(h0 * self.Fy / fy0)
        w1 = int(w0 * self.Fx / fx0)

        scale_factor_yx = (float(h1) / h0, float(w1) / w0)

        # resize
        left_image = cv2.resize(left_image, (w1, h1))
        if right_image is not None:
            right_image = cv2.resize(right_image, (w1, h1))
        if image_gt is not None:
            image_gt = cv2.resize(image_gt, (w1, h1), cv2.INTER_NEAREST)

        if p2 is not None:
            p2[0, :]   = p2[0, :] * scale_factor_yx[1]
            p2[1, :]   = p2[1, :] * scale_factor_yx[0]
        
        if p3 is not None:
            p3[0, :]   = p3[0, :] * scale_factor_yx[1]
            p3[1, :]   = p3[1, :] * scale_factor_yx[0]
        
        if labels:
            if isinstance(labels, list):
                for obj in labels:
                    obj.bbox_l *= scale_factor_yx[1]
                    obj.bbox_r *= scale_factor_yx[1]
                    obj.bbox_t *= scale_factor_yx[0]
                    obj.bbox_b *= scale_factor_yx[0]

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomSaturation(object):
    """
    Randomly adjust the saturation of an image given a lower and upper bound,
    and a distortion probability.

    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5):

        self.distort_prob = distort_prob
        self.lower = lower
        self.upper = upper

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if random.rand() <= self.distort_prob:
            ratio = random.uniform(self.lower, self.upper)
            left_image[:, :, 1] *= ratio
            if right_image is not None:
                right_image[:, :, 1] *= ratio

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class CropTop(object):
    def __init__(self, crop_top_index=None, output_height=None):
        if crop_top_index is None and output_height is None:
            print("Either crop_top_index or output_height should not be None, set crop_top_index=0 by default")
            crop_top_index = 0
        if crop_top_index is not None and output_height is not None:
            print("Neither crop_top_index or output_height is None, crop_top_index will take over")
        self.crop_top_index = crop_top_index
        self.output_height = output_height

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        height, width = left_image.shape[0:2]

        if self.crop_top_index is not None:
            h_out = height - self.crop_top_index
            upper = self.crop_top_index
        else:
            h_out = self.output_height
            upper = height - self.output_height
        lower = height

        left_image = left_image[upper:lower]
        if right_image is not None:
            right_image = right_image[upper:lower]
        if image_gt is not None:
            image_gt = image_gt[upper:lower]
        ## modify calibration matrix
        if p2 is not None:
            p2[1, 2] = p2[1, 2] - upper               # cy' = cy - dv
            p2[1, 3] = p2[1, 3] - upper * p2[2, 3] # ty' = ty - dv * tz

        if p3 is not None:
            p3[1, 2] = p3[1, 2] - upper               # cy' = cy - dv
            p3[1, 3] = p3[1, 3] - upper * p3[2, 3] # ty' = ty - dv * tz

        
        if labels is not None:
            if isinstance(labels, list):
                # scale all coordinates
                for obj in labels:
                    obj.bbox_b -= upper
                    obj.bbox_t -= upper

        return left_image, right_image, p2, p3, labels, image_gt, lidar


@AUGMENTATION_DICT.register_module
class CropRight(object):
    def __init__(self, crop_right_index=None, output_width=None):
        if crop_right_index is None and output_width is None:
            print("Either crop_right_index or output_width should not be None, set crop_right_index=0 by default")
            crop_right_index = 0
        if crop_right_index is not None and output_width is not None:
            print("Neither crop_right_index or output_width is None, crop_right_index will take over")
        self.crop_right_index = crop_right_index
        self.output_width = output_width

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        height, width = left_image.shape[0:2]

        lefter = 0
        if self.crop_right_index is not None:
            w_out = width - self.crop_right_index
            righter = w_out
        else:
            w_out = self.output_width
            righter = w_out
        
        if righter > width:
            print("does not crop right since it is larger")
            return left_image, right_image, p2, p3, labels

        # crop left image
        left_image = left_image[:, lefter:righter, :]

        # crop right image if possible
        if right_image is not None:
            right_image = right_image[:, lefter:righter, :]

        if image_gt is not None:
            image_gt = image_gt[:, lefter:righter]

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class FilterObject(object):
    """
        Filtering out object completely outside of the box;
    """
    def __init__(self):
        pass

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        height, width = left_image.shape[0:2]

        if labels is not None:
            new_labels = []
            if isinstance(labels, list):
                # scale all coordinates
                for obj in labels:
                    is_outside = (
                        obj.bbox_b < 0 or obj.bbox_t > height or obj.bbox_r < 0 or obj.bbox_l > width
                    )
                    if not is_outside:
                        new_labels.append(obj)
        else:
            new_labels = None
        
        return left_image, right_image, p2, p3, new_labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomCropToWidth(object):
    def __init__(self, width:int):
        self.width = width

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        height, original_width = left_image.shape[0:2]

        

        if self.width > original_width:
            print("does not crop since it is larger")
            return left_image, right_image, p2, p3, labels, image_gt


        lefter = np.random.randint(0, original_width - self.width)
        righter = lefter + self.width
        # crop left image
        left_image = left_image[:, lefter:righter, :]

        # crop right image if possible
        if right_image is not None:
            right_image = right_image[:, lefter:righter, :]

        if image_gt is not None:
            image_gt = image_gt[:, lefter:righter]

        ## modify calibration matrix
        if p2 is not None:
            p2[0, 2] = p2[0, 2] - lefter               # cy' = cy - dv
            p2[0, 3] = p2[0, 3] - lefter * p2[2, 3] # ty' = ty - dv * tz

        if p3 is not None:
            p3[0, 2] = p3[0, 2] - lefter               # cy' = cy - dv
            p3[0, 3] = p3[0, 3] - lefter * p3[2, 3] # ty' = ty - dv * tz

        
        if labels:
            # scale all coordinates
            if isinstance(labels, list):
                for obj in labels:
                        
                    obj.bbox_l -= lefter
                    obj.bbox_r -= lefter

            

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomMirror(object):
    """
    Randomly mirror an image horzontially, given a mirror probabilty.

    Also, adjust all box cordinates accordingly.
    """
    def __init__(self, mirror_prob):
        self.mirror_prob = mirror_prob
        self.projector = BBox3dProjector()

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):

        _, width, _ = left_image.shape

        if random.rand() <= self.mirror_prob:

            left_image = left_image[:, ::-1, :]
            left_image = np.ascontiguousarray(left_image)

            if right_image is not None:
                right_image = right_image[:, ::-1, :]
                right_image = np.ascontiguousarray(right_image)

                left_image, right_image = right_image, left_image
            if image_gt is not None:
                image_gt = image_gt[:, ::-1]
                image_gt = np.ascontiguousarray(image_gt)

            # flip the coordinates w.r.t the horizontal flip (only adjust X)
            if p2 is not None and p3 is not None:
                p2, p3 = p3, p2
            if p2 is not None:
                p2[0, 3] = -p2[0, 3]
                p2[0, 2] = left_image.shape[1] - p2[0, 2] - 1
            if p3 is not None:
                p3[0, 3] = -p3[0, 3]
                p3[0, 2] = left_image.shape[1] - p3[0, 2] - 1
            if labels:
                if isinstance(labels, list):
                    square_P2 = np.eye(4)
                    square_P2[0:3, :] = p2
                    p2_inv = np.linalg.inv(square_P2)
                    for obj in labels:
                        # In stereo horizontal 2D boxes will be fixed later when we use 3D projection as 2D anchor box
                        obj.bbox_l, obj.bbox_r = left_image.shape[1] - obj.bbox_r - 1, left_image.shape[1] - obj.bbox_l - 1
                        
                        # 3D centers
                        z = obj.z
                        obj.x = -obj.x

                        # yaw
                        ry = obj.ry
                        ry = (-math.pi - ry) if ry < 0 else (math.pi - ry)
                        while ry > math.pi: ry -= math.pi * 2
                        while ry < (-math.pi): ry += math.pi * 2
                        obj.ry = ry

                        # alpha 
                        obj.alpha = theta2alpha_3d(ry, obj.x, z, p2)
                
            if lidar is not None:
                lidar[:, :, 0] = -lidar[:, :, 0]
        
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomWarpAffine(object):
    """
        Randomly random scale and random shift the image. Then resize to a fixed output size. 
    """
    def __init__(self, scale_lower=0.6, scale_upper=1.4, shift_border=128, output_w=1280, output_h=384):
        self.scale_lower    = scale_lower
        self.scale_upper    = scale_upper
        self.shift_border   = shift_border
        self.output_w       = output_w
        self.output_h       = output_h

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        s_original = max(left_image.shape[0], left_image.shape[1])
        center_original = np.array([left_image.shape[1] / 2., left_image.shape[0] / 2.], dtype=np.float32)
        scale = s_original * np.random.uniform(self.scale_lower, self.scale_upper)
        center_w = np.random.randint(low=self.shift_border, high=left_image.shape[1] - self.shift_border)
        center_h = np.random.randint(low=self.shift_border, high=left_image.shape[0] - self.shift_border)

        final_scale = max(self.output_w, self.output_h) / scale
        final_shift_w = self.output_w / 2 - center_w * final_scale
        final_shift_h = self.output_h / 2 - center_h * final_scale
        affine_transform = np.array(
            [
                [final_scale, 0, final_shift_w],
                [0, final_scale, final_shift_h]
            ], dtype=np.float32
        )

        left_image = cv2.warpAffine(left_image, affine_transform,
                                    (self.output_w, self.output_h), flags=cv2.INTER_LINEAR)
        if right_image is not None:
            right_image = cv2.warpAffine(right_image, affine_transform,
                                    (self.output_w, self.output_h), flags=cv2.INTER_LINEAR)

        if image_gt is not None:
            image_gt = cv2.warpAffine(image_gt, affine_transform,
                                    (self.output_w, self.output_h), flags=cv2.INTER_LINEAR)

        if p2 is not None:
            p2[0:2, :] *= final_scale
            p2[0, 2] = p2[0, 2] + final_shift_w               # cy' = cy - dv
            p2[0, 3] = p2[0, 3] + final_shift_w * p2[2, 3] # ty' = ty - dv * tz
            p2[1, 2] = p2[1, 2] + final_shift_h               # cy' = cy - dv
            p2[1, 3] = p2[1, 3] + final_shift_h * p2[2, 3] # ty' = ty - dv * tz

        if p3 is not None:
            p3[0:2, :] *= final_scale
            p3[0, 2] = p3[0, 2] + final_shift_w               # cy' = cy - dv
            p3[0, 3] = p3[0, 3] + final_shift_w * p3[2, 3] # ty' = ty - dv * tz
            p3[1, 2] = p3[1, 2] + final_shift_h               # cy' = cy - dv
            p3[1, 3] = p3[1, 3] + final_shift_h * p3[2, 3] # ty' = ty - dv * tz
        
        if labels:
            if isinstance(labels, list):
                for obj in labels:
                    obj.bbox_l = obj.bbox_l * final_scale + final_shift_w
                    obj.bbox_r = obj.bbox_r * final_scale + final_shift_w
                    obj.bbox_t = obj.bbox_t * final_scale + final_shift_h
                    obj.bbox_b = obj.bbox_b * final_scale + final_shift_h

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomHue(object):
    """
    Randomly adjust the hue of an image given a delta degree to rotate by,
    and a distortion probability.

    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        self.distort_prob = distort_prob

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if random.rand() <= self.distort_prob:
            shift = random.uniform(-self.delta, self.delta)
            left_image[:, :, 0] += shift
            left_image[:, :, 0][left_image[:, :, 0] > 360.0] -= 360.0
            left_image[:, :, 0][left_image[:, :, 0] < 0.0] += 360.0
            if right_image is not None:
                right_image[:, :, 0] += shift
                right_image[:, :, 0][right_image[:, :, 0] > 360.0] -= 360.0
                right_image[:, :, 0][right_image[:, :, 0] < 0.0] += 360.0
        return left_image, right_image, p2, p3, labels, image_gt, lidar


@AUGMENTATION_DICT.register_module
class ConvertColor(object):
    """
    Converts color spaces to/from HSV and RGB
    """
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):

        # RGB --> HSV
        if self.current == 'RGB' and self.transform == 'HSV':
            left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2HSV)
            if right_image is not None:
                right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2HSV)

        # HSV --> RGB
        elif self.current == 'HSV' and self.transform == 'RGB':
            left_image = cv2.cvtColor(left_image, cv2.COLOR_HSV2RGB)
            if right_image is not None:
                right_image = cv2.cvtColor(right_image, cv2.COLOR_HSV2RGB)

        else:
            raise NotImplementedError

        return left_image, right_image, p2, p3, labels, image_gt, lidar


@AUGMENTATION_DICT.register_module
class RandomContrast(object):
    """
    Randomly adjust contrast of an image given lower and upper bound,
    and a distortion probability.
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5):

        self.lower = lower
        self.upper = upper
        self.distort_prob = distort_prob

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if random.rand() <= self.distort_prob:
            alpha = random.uniform(self.lower, self.upper)
            left_image *= alpha
            if right_image is not None:
                right_image *= alpha
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomBrightness(object):
    """
    Randomly adjust the brightness of an image given given a +- delta range,
    and a distortion probability.
    """
    def __init__(self, distort_prob, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.distort_prob = distort_prob

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if random.rand() <= self.distort_prob:
            delta = random.uniform(-self.delta, self.delta)
            left_image += delta
            if right_image is not None:
                right_image += delta
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomEigenvalueNoise(object):
    """
        Randomly apply noise in RGB color channels based on the eigenvalue and eigenvector of ImageNet
    """
    def __init__(self, distort_prob=1.0,
                       alphastd=0.1,
                       eigen_value=np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32),
                       eigen_vector=np.array([
                            [-0.58752847, -0.69563484, 0.41340352],
                            [-0.5832747, 0.00994535, -0.81221408],
                            [-0.56089297, 0.71832671, 0.41158938]
                        ], dtype=np.float32)
                ):
        self.distort_prob = distort_prob
        self._eig_val = eigen_value
        self._eig_vec = eigen_vector
        self.alphastd = alphastd

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if random.rand() <= self.distort_prob:
            alpha = np.random.normal(scale=self.alphastd, size=(3, ))
            noise = np.dot(self._eig_vec, self._eig_val * alpha) * 255

            left_image += noise
            if right_image is not None:
                right_image += noise
            
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class PhotometricDistort(object):
    """
    Packages all photometric distortions into a single transform.
    """
    def __init__(self, distort_prob=1.0, contrast_lower=0.5, contrast_upper=1.5, saturation_lower=0.5, saturation_upper=1.5, hue_delta=18.0, brightness_delta=32):

        self.distort_prob = distort_prob

        # contrast is duplicated because it may happen before or after
        # the other transforms with equal probability.
        self.transforms = [
            RandomContrast(distort_prob, contrast_lower, contrast_upper),
            ConvertColor(transform='HSV'),
            RandomSaturation(distort_prob, saturation_lower, saturation_upper),
            RandomHue(distort_prob, hue_delta),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast(distort_prob, contrast_lower, contrast_upper)
        ]

        self.rand_brightness = RandomBrightness(distort_prob, brightness_delta)

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):

        # do contrast first
        if random.rand() <= 0.5:
            distortion = self.transforms[:-1]

        # do contrast last
        else:
            distortion = self.transforms[1:]

        # add random brightness
        distortion.insert(0, self.rand_brightness)

        # compose transformation
        distortion = Compose.from_transforms(distortion)

        return distortion(left_image.copy(), right_image if right_image is None else right_image.copy(), p2, p3, labels, image_gt, lidar)


class Augmentation(object):
    """
    Data Augmentation class which packages the typical pre-processing
    and all data augmentation transformations (mirror and photometric distort)
    into a single transform.
    """
    def __init__(self, cfg):

        self.mean = cfg.rgb_mean
        self.stds = cfg.rgb_std
        self.size = cfg.cropSize
        self.mirror_prob = cfg.mirrorProb
        self.distort_prob = cfg.distortProb

        if cfg.distortProb <= 0:
            self.augment = Compose.from_transforms([
                ConvertToFloat(),
                CropTop(cfg.crop_top),
                Resize(self.size),
                RandomMirror(self.mirror_prob),
                Normalize(self.mean, self.stds)
            ])
        else:
            self.augment = Compose.from_transforms([
                ConvertToFloat(),
                PhotometricDistort(self.distort_prob),
                CropTop(cfg.crop_top),
                #RandomCrop(self.distort_prob, self.size),
                Resize(self.size),
                RandomMirror(self.mirror_prob),
                Normalize(self.mean, self.stds)
            ])

    def __call__(self, left_image, right_image, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        return self.augment(left_image, right_image, p2, p3, labels, image_gt, lidar)


class Preprocess(object):
    """
    Preprocess function which ONLY does the basic pre-processing of an image,
    meant to be used during the testing/eval stages.
    """
    def __init__(self, cfg):

        self.mean = cfg.rgb_mean
        self.stds = cfg.rgb_std
        self.size = cfg.cropSize

        self.preprocess = Compose.from_transforms([
            ConvertToFloat(),
            CropTop(cfg.crop_top),
            Resize(self.size),
            Normalize(self.mean, self.stds)
        ])

    def __call__(self, left_image, right_image, p2=None, p3=None, labels=None, image_gt=None, lidar=None):

        left_image, right_image, p2, p3, labels, image_gt, lidar = self.preprocess(left_image, right_image, p2, p3, labels, image_gt, lidar)

        #img = np.transpose(img, [2, 0, 1])

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class Shuffle(object):
    """
        Initialize a sequence of transformations. During function call, it will randomly shuffle the augmentation calls.

        Can be used with Compose to build complex augmentation structures.
    """
    def __init__(self, aug_list:List[EasyDict]):
        self.transforms = [
            build_single_augmentator(aug_cfg) for aug_cfg in aug_list
        ]

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        # We aim to keep the original order of the initialized transforms in self.transforms, so we only randomize the indexes.
        shuffled_indexes = np.random.permutation(len(self.transforms))

        for index in shuffled_indexes:
            left_image, right_image, p2, p3, labels, image_gt, lidar = self.transforms[index](left_image, right_image, p2, p3, labels, image_gt, lidar)
        
        return left_image, right_image, p2, p3, labels, image_gt, lidar

