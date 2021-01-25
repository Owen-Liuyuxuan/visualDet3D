# data package

## pipeline

We use a registry dictionary mechanism following mmdetection. One could build monocular/stereo augmentator with a list of dictionaries.

## kitti

### kittidata/utils
Contain object type definition for object data following det3, while adding support for stereo.

Numba accelerated utils for data reading and coordinate transforms.

### dataset
pytorch dataset implementation for monocular/stereo/depth_prediction and more.

### *_split
split files for train/val/test

