from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import os
class Anchors(nn.Module):
    """ Anchor modules for multi-level dense output.
    
    """
    def __init__(self, preprocessed_path:str,
                       pyramid_levels:List[int], strides:List[float], sizes:List[float], ratios:List[float], scales:List[float],
                       readConfigFile:int=1, obj_types:List[str]=[],
                       filter_anchors:bool=True, filter_y_threshold_min_max:Optional[Tuple[float, float]]=(-0.5, 1.8), filter_x_threshold:Optional[float]=40.0,
                       anchor_prior_channel=6):
        super(Anchors, self).__init__()

        self.pyramid_levels = pyramid_levels

        self.strides = strides

        self.sizes = sizes

        self.ratios = ratios

        self.scales = scales
        self.shape = None
        self.P2 = None
        self.readConfigFile = readConfigFile
        self.scale_step = 1 / (np.log2(self.scales[1]) - np.log2(self.scales[0]))
        if self.readConfigFile:
            self.anchors_mean_original = np.zeros([len(obj_types), len(self.scales) * len(self.pyramid_levels), len(self.ratios), anchor_prior_channel])
            self.anchors_std_original  = np.zeros([len(obj_types), len(self.scales) * len(self.pyramid_levels), len(self.ratios), anchor_prior_channel])
            save_dir = os.path.join(preprocessed_path, 'training')
            for i in range(len(obj_types)):
                npy_file = os.path.join(save_dir,'anchor_mean_{}.npy'.format(obj_types[i]))
                self.anchors_mean_original[i]  = np.load(npy_file) #[30, 2, 6] #[z,  sinalpha, cosalpha, w, h, l,]
                

                std_file = os.path.join(save_dir,'anchor_std_{}.npy'.format(obj_types[i]))
                self.anchors_std_original[i] = np.load(std_file) #[30, 2, 6] #[z,  sinalpha, cosalpha, w, h, l,]

        self.filter_y_threshold_min_max = filter_y_threshold_min_max
        self.filter_x_threshold = filter_x_threshold

    def anchors2indexes(self, anchors:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        """
            computations in numpy: anchors[N, 4]
            return: sizes_int [N,]  ratio_ints [N, ]
        """
        sizes = np.sqrt((anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1]))
        sizes_diff = sizes - (np.array(self.sizes) * np.array(self.scales))[:, np.newaxis]
        sizes_int = np.argmin(np.abs(sizes_diff), axis=0)

        ratio =  (anchors[:, 3] - anchors[:, 1]) / (anchors[:, 2] - anchors[:, 0])
        ratio_diff = ratio - np.array(self.ratios)[:, np.newaxis]
        ratio_int = np.argmin(np.abs(ratio_diff), axis=0)
        return sizes_int, ratio_int

    def forward(self, image:torch.Tensor, calibs:List[np.ndarray]=[], is_filtering=False):
        shape = image.shape[2:]
        if self.shape is None or not (shape == self.shape):
            self.shape = image.shape[2:]
            
            image_shape = image.shape[2:]
            image_shape = np.array(image_shape)
            image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

            # compute anchors over all pyramid levels
            all_anchors = np.zeros((0, 4)).astype(np.float32)

            for idx, p in enumerate(self.pyramid_levels):
                anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
                shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
                all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

            if self.readConfigFile:
                sizes_int, ratio_int = self.anchors2indexes(all_anchors)
                self.anchor_means = image.new(self.anchors_mean_original[:, sizes_int, ratio_int]) #[types, N, 6]
                self.anchor_stds  = image.new(self.anchors_std_original[:, sizes_int, ratio_int]) #[types, N, 6]
                self.anchor_mean_std = torch.stack([self.anchor_means, self.anchor_stds], dim=-1).permute(1, 0, 2, 3) #[N, types, 6, 2]

            all_anchors = np.expand_dims(all_anchors, axis=0)
            if isinstance(image, torch.Tensor):
                self.anchors = image.new(all_anchors.astype(np.float32)) #[1, N, 4]
            elif isinstance(image, np.ndarray):
                self.anchors = torch.tensor(all_anchors.astype(np.float32)).cuda()
            self.anchors_image_x_center = self.anchors[0,:,0:4:2].mean(dim=1) #[N]
            self.anchors_image_y_center = self.anchors[0,:,1:4:2].mean(dim=1) #[N]

        if calibs is not None and len(calibs) > 0:
            #P2 = calibs.P2 #[3, 4]
            #P2 = np.stack([calib for calib in calibs]) #[B, 3, 4]
            P2 = calibs #[B, 3, 4]
            if self.P2 is not None and torch.all(self.P2 == P2) and self.P2.shape == P2.shape:
                if self.readConfigFile:
                    return self.anchors, self.useful_mask, self.anchor_mean_std
                else:
                    return self.anchors, self.useful_mask

            self.P2 = P2
            fy = P2[:, 1:2, 1:2] #[B,1, 1]
            cy = P2[:, 1:2, 2:3] #[B,1, 1]
            cx = P2[:, 0:1, 2:3] #[B,1, 1]
            N = self.anchors.shape[1]
            if self.readConfigFile and is_filtering:
                anchors_z = self.anchor_means[:, :, 0] #[types, N]
                world_x3d = (self.anchors_image_x_center * anchors_z - anchors_z.new(cx) * anchors_z) / anchors_z.new(fy) #[B, types, N]
                world_y3d = (self.anchors_image_y_center * anchors_z - anchors_z.new(cy) * anchors_z) / anchors_z.new(fy) #[B, types, N]
                self.useful_mask = torch.any( (world_y3d > self.filter_y_threshold_min_max[0]) * 
                                              (world_y3d < self.filter_y_threshold_min_max[1]) *
                                              (world_x3d.abs() < self.filter_x_threshold), dim=1)  #[B,N] any one type lies in target range
            else:
                self.useful_mask = torch.ones([len(P2), N], dtype=torch.bool, device="cuda")
            if self.readConfigFile:
                return self.anchors, self.useful_mask, self.anchor_mean_std
            else:
                return self.anchors, self.useful_mask
        return self.anchors

    @property
    def num_anchors(self):
        return len(self.pyramid_levels) * len(self.ratios) * len(self.scales)

    @property
    def num_anchor_per_scale(self):
        return len(self.ratios) * len(self.scales)

    @staticmethod
    def _deshift_anchors(anchors):
        """shift the anchors to zero base

        Args:
            anchors: [..., 4] [x1, y1, x2, y2]
        Returns:
            [..., 4] [x1, y1, x2, y2] as with (x1 + x2) == 0 and (y1 + y2) == 0
        """
        x1 = anchors[..., 0]
        y1 = anchors[..., 1]
        x2 = anchors[..., 2]
        y2 = anchors[..., 3]
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)

        return torch.stack([
            x1 - center_x,
            y1 - center_y,
            x2 - center_x,
            y2 - center_y
        ], dim=-1)
        

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

