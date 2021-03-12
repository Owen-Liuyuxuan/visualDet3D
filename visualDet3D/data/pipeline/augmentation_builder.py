from typing import Callable, List, Union
import numpy as np
from easydict import EasyDict
from visualDet3D.networks.utils.registry import AUGMENTATION_DICT
from visualDet3D.data.kitti.kittidata import KittiObj

def build_single_augmentator(cfg:EasyDict):
    name:str = cfg.type_name
    keywords:dict = getattr(cfg, 'keywords', dict())
    return AUGMENTATION_DICT[name](**keywords)

@AUGMENTATION_DICT.register_module
class Compose(object):
    """
    Composes a set of functions which take in an image and an object, into a single transform
    """
    # def __init__(self, transforms:List[Callable], is_return_all=True):
    #     self.transforms = transforms
    #     self.is_return_all = is_return_all

    def __init__(self, aug_list:List[EasyDict], is_return_all=True):
        self.transforms:List[Callable] = []
        for item in aug_list:
            self.transforms.append(build_single_augmentator(item))
        self.is_return_all = is_return_all

    @classmethod
    def from_transforms(cls, transforms:List[Callable]): 
        instance:Compose = cls(aug_list=[])
        instance.transforms = transforms
        return instance

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


def build_augmentator(aug_cfg:List[EasyDict])->Compose:
    return Compose(aug_cfg, is_return_all=False)
