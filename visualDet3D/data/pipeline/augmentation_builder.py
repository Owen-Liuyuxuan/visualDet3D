from typing import Callable, List, Union
import numpy as np
from easydict import EasyDict
from visualDet3D.networks.utils.registry import AUGMENTATION_DICT
from visualDet3D.data.kitti.kittidata import KittiObj
class Compose(object):
    """
    Composes a set of functions which take in an image and an object, into a single transform
    """
    def __init__(self, transforms:List[Callable], is_return_all=True):
        self.transforms = transforms
        self.is_return_all = is_return_all

    def __call__(self, left_image:np.ndarray,
                       right_image:Union[None, np.ndarray]=None,
                       p2:Union[None, np.ndarray]=None,
                       p3:Union[None, np.ndarray]=None,
                       labels:Union[None, List[KittiObj]]=None,
                       image_gt:Union[None, np.ndarray]=None,
                       lidar:Union[None, np.ndarray]=None)->List[Union[None, np.ndarray, List[KittiObj]]]:
        """
            if self.is_return_all:
                The return list will follow the common signature: left_image, right_image, p2, p3, labels, image_gt, lidar(in the camera coordinate)
                Mainly used in composing a group of augmentator into one complex one.
            else:
                The return list will follow the input argument; only return items that is not None in the input.
                Used in the final wrapper to provide a more flexible interface.
        """
        for t in self.transforms:
            left_image, right_image, p2, p3, labels, image_gt, lidar = t(left_image, right_image, p2, p3, labels, image_gt, lidar)
        return_list = [left_image, right_image, p2, p3, labels, image_gt, lidar]
        if self.is_return_all:
            return return_list
        return [item for item in return_list if item is not None]


def build_augmentator(aug_cfg:EasyDict)->Compose:
    transformers:List[Callable] = []
    for item in aug_cfg:
        name = item.type_name
        keywords = getattr(item, 'keywords', dict())
        transformers.append(
            AUGMENTATION_DICT[name](**keywords)
        )
    return Compose(transformers, is_return_all=False)
