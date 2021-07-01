import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.ops import nms
from easydict import EasyDict
import numpy as np

from visualDet3D.networks.heads.losses import IoULoss
from visualDet3D.networks.utils.rtm3d_utils import _transpose_and_gather_feat,\
      _nms, _topk, decode_depth_inv_sigmoid, decode_depth_from_keypoints
from .km3d_head import KM3DHead


class MonoFlexHead(KM3DHead):

    def build_loss(self,
                   uncertainty_range=[-10, 10],
                   uncertainty_weight=1.0,
                   **kwargs):
        assert uncertainty_range[1] >= uncertainty_range[0]    
        self.bbox2d_loss = IoULoss()
        self.uncertainty_range = uncertainty_range
        self.uncertainty_weight = uncertainty_weight

    def _bbox2d_loss(self, output:torch.Tensor,
                           target:torch.Tensor)->torch.Tensor:
        pred_box = torch.cat([output[..., 0:2] * -1, output[..., 2:]], dim=-1)
        targ_box = torch.cat([target[..., 0:2] * -1, target[..., 2:]], dim=-1)
        loss = self.bbox2d_loss(pred_box, targ_box).sum()
        loss = loss / (len(output) + 1e-4)
        return loss

    def _laplacian_l1(self, output, target, uncertainty):
        loss = F.l1_loss(output, target, reduction='none') * torch.exp(-uncertainty) + \
            uncertainty * self.uncertainty_weight
        
        return loss.sum() / (len(output) + 1e-4)
    
    @staticmethod
    def _L1Loss(output, target):
        loss = F.l1_loss(output, target, reduction='none')
        return loss.sum() / (len(output) + 1e-4)

    def _gather_output(self, output, ind, mask):

        bbox2d = _transpose_and_gather_feat(output['bbox2d'], ind)[mask]
        dim = _transpose_and_gather_feat(output['dim'], ind)[mask]
        rot = _transpose_and_gather_feat(output['rot'], ind)[mask]
        hps = _transpose_and_gather_feat(output['hps'], ind)[mask]
        offset = _transpose_and_gather_feat(output['reg'], ind)[mask]
        depth = _transpose_and_gather_feat(output['depth'], ind)[mask]
        dim = _transpose_and_gather_feat(output['dim'], ind)[mask]
        depth_uncer = _transpose_and_gather_feat(output['depth_uncertainty'], ind)[mask]
        corner_uncer = _transpose_and_gather_feat(output['corner_uncertainty'], ind)[mask]

        n, _ = hps.shape #[N, 20]
        hps = hps.reshape(n, -1, 2) #[N, 10, 2]

        flatten_reg_mask_gt = mask.view(-1).bool()
        batch_idxs = torch.arange(len(mask)).view(-1, 1).expand_as(mask).reshape(-1)
        batch_idxs = batch_idxs[flatten_reg_mask_gt].long().cuda()

        decoded_dict = dict(
            bbox2d = bbox2d,
            dim=dim,
            rot=rot,
            hps=hps,
            offset = offset,
            depth=depth,
            depth_uncer=depth_uncer,
            corner_uncer=corner_uncer,
            batch_idxs=batch_idxs
        )
        return decoded_dict

    def _keypoints_depth_loss(self, depths, target, validmask, uncertainty):

        loss = F.l1_loss(depths, target.repeat(1, 3), reduction='none') * torch.exp(-uncertainty) + \
            uncertainty * self.uncertainty_weight #[N, 3]
        
        valid_loss = loss * validmask.float() + (1 - validmask.float()) * loss.detach() #[N, 3]

        return valid_loss.mean(dim=1).sum() / (len(depths) + 1e-4)

    @staticmethod
    def merge_depth(depth, depth_uncer):
        pred_uncertainty_weights = 1 / depth_uncer
        pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
        depth = torch.sum(depth * pred_uncertainty_weights, dim=1)
        return depth
    
    def _decode(self, reg_preds, points):
        xs = points[:, 0] #[N]
        ys = points[:, 1] #[N]

        lefts   = xs - reg_preds[..., 0] #[N, ]
        tops    = ys - reg_preds[..., 1] #[N, ]
        rights  = xs + reg_preds[..., 2] #[N, ]
        bottoms = ys + reg_preds[..., 3] #[N, ]

        bboxes = torch.stack([lefts, tops, rights, bottoms], dim=-1)

        return bboxes

    def _decode_alpha(self, rot):
        alpha_idx = rot[..., 1] > rot[..., 5]
        alpha_idx = alpha_idx.float()
        alpha1 = torch.atan(rot[..., 2] / rot[..., 3]) + (-0.5 * np.pi)
        alpha2 = torch.atan(rot[..., 6] / rot[..., 7]) + (0.5 * np.pi)
        alpha_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
        return alpha_pre

    def get_bboxes(self, output:dict, P2, img_batch=None):
        output['hm'] = torch.sigmoid(output['hm'])

        heat = _nms(output['hm'])
        scores, inds, clses, ys, xs = _topk(heat, K=100)

        gathered_output = self._gather_output(output, inds.long(), torch.ones_like(scores).bool())

        scores = scores[0] #[1, N] -> [N]
        clses = clses[0] #[1, N] -> [N]
        ys = ys[0] #[1, N] -> [N]
        xs = xs[0] #[1, N] -> [N]

        bbox2d = self._decode(gathered_output['bbox2d'], torch.stack([xs, ys], dim=1))

        gathered_output['depth_decoded'] = decode_depth_inv_sigmoid(gathered_output['depth'])
        expanded_P2 = P2[gathered_output['batch_idxs'], :, :] #[N, 4, 4]
        gathered_output['kpd_depth'] = decode_depth_from_keypoints(gathered_output['hps'], gathered_output['dim'], expanded_P2) #[N, 3]
        gathered_output['depth_uncer'] = torch.clamp(gathered_output['depth_uncer'], self.uncertainty_range[0], self.uncertainty_range[1])
        gathered_output['corner_uncer'] = torch.clamp(gathered_output['corner_uncer'], self.uncertainty_range[0], self.uncertainty_range[1])

        pred_combined_uncertainty = torch.cat((gathered_output['depth_uncer'], gathered_output['corner_uncer']), dim=1).exp()
        pred_combined_depths = torch.cat((gathered_output['depth_decoded'], gathered_output['kpd_depth']), dim=1)
        gathered_output['merged_depth'] = self.merge_depth(pred_combined_depths, pred_combined_uncertainty)

        score_threshold = getattr(self.test_cfg, 'score_thr', 0.1)
        mask = scores > score_threshold#[K]
        bbox2d = bbox2d[mask]
        scores = scores[mask].unsqueeze(1) #[K, 1]
        dims   = gathered_output['dim'][mask] #[w, h, l] ? 
        cls_indexes = clses[mask].long()
        alpha = self._decode_alpha(gathered_output['rot'][mask]).unsqueeze(-1)

        cx3d = (xs[mask] + gathered_output['offset'][mask][..., 0]).unsqueeze(-1)
        cy3d = (ys[mask] + gathered_output['offset'][mask][..., 1]).unsqueeze(-1)
        z3d = gathered_output['merged_depth'][mask].unsqueeze(-1)  #[N, 1]
    
        ## upsample back
        bbox2d *= 4
        cx3d *= 4
        cy3d *= 4

        if img_batch is not None:
            bbox2d = self.clipper(bbox2d, img_batch)

        bbox3d_3d = torch.cat(
            [bbox2d,  cx3d, cy3d, z3d, dims, alpha], dim=1                  #cx, cy, z, w, h, l, alpha
        )
        
        cls_agnostic = getattr(self.test_cfg, 'cls_agnositc', True) # True -> directly NMS; False -> NMS with offsets different categories will not collide
        nms_iou_thr  = getattr(self.test_cfg, 'nms_iou_thr', 0.5)
        

        if cls_agnostic:
            keep_inds = nms(bbox3d_3d[:, :4], scores[:, 0], nms_iou_thr)
        else:
            max_coordinate = bbox3d_3d.max()
            nms_bbox = bbox3d_3d[:, :4] + cls_indexes.float() * (max_coordinate)
            keep_inds = nms(nms_bbox, scores, nms_iou_thr)
            
        scores = scores[keep_inds, 0]
        bbox3d_3d = bbox3d_3d[keep_inds]
        cls_indexes = cls_indexes[keep_inds]
        

        return scores, bbox3d_3d, cls_indexes

    def loss(self, output, annotations, meta):
        P2 = meta['P2']
        epoch = meta['epoch']

        annotations['ind'] = annotations['ind'].long()
        annotations['reg_mask'] = annotations['reg_mask'].bool()
        # heatmap center loss
        hm_loss = self._neg_loss(output['hm'], annotations['hm'])
        # keypoint L1 loss
        hp_loss = self._RegWeightedL1Loss(output['hps'],annotations['hps_mask'], annotations['ind'], annotations['hps'],annotations['dep'].clone())
        # rotations from RTM3D
        rot_loss = self._RotLoss(output['rot'], annotations['reg_mask'], annotations['ind'], annotations['rotbin'], annotations['rotres'])

        # gather output
        gathered_output = self._gather_output(output, annotations['ind'], annotations['reg_mask'])
        gathered_output['depth_decoded'] = decode_depth_inv_sigmoid(gathered_output['depth'])
        expanded_P2 = P2[gathered_output['batch_idxs'], :, :] #[N, 3, 4]
        gathered_output['kpd_depth'] = decode_depth_from_keypoints(gathered_output['hps'], gathered_output['dim'], expanded_P2) #[N, 3]
        gathered_output['depth_uncer'] = torch.clamp(gathered_output['depth_uncer'], self.uncertainty_range[0], self.uncertainty_range[1])
        gathered_output['corner_uncer'] = torch.clamp(gathered_output['corner_uncer'], self.uncertainty_range[0], self.uncertainty_range[1])

        pred_combined_uncertainty = torch.cat((gathered_output['depth_uncer'], gathered_output['corner_uncer']), dim=1).exp()
        pred_combined_depths = torch.cat((gathered_output['depth_decoded'], gathered_output['kpd_depth']), dim=1)
        gathered_output['merged_depth'] = self.merge_depth(pred_combined_depths, pred_combined_uncertainty)

        # FCOS style regression
        box2d_loss = self._bbox2d_loss(gathered_output['bbox2d'], annotations['bboxes2d_target'][annotations['reg_mask']])
        # dimensions
        dim_loss = self._L1Loss(gathered_output['dim'], annotations['dim'][annotations['reg_mask']])
        # offset for center heatmap
        off_loss = self._L1Loss(gathered_output['offset'], annotations['reg'][annotations['reg_mask']])

        # direct depth regression
        depth_loss = self._laplacian_l1(gathered_output['depth_decoded'], annotations['dep'][annotations['reg_mask']], gathered_output['depth_uncer'])

        keypoint_depth_loss = self._keypoints_depth_loss(gathered_output['kpd_depth'], annotations['dep'][annotations['reg_mask']],
                                                         annotations['kp_detph_mask'][annotations['reg_mask']], gathered_output['corner_uncer'])
        soft_depth_loss = self._L1Loss(gathered_output['merged_depth'].unsqueeze(-1), annotations['dep'][annotations['reg_mask']])


        loss_stats = {'hm_loss': hm_loss, 'hp_loss': hp_loss,
                      'box2d_loss': box2d_loss, 'off_loss': off_loss,'dim_loss': dim_loss,
                      'depth_loss': depth_loss, 'kpd_loss': keypoint_depth_loss,
                      'rot_loss':rot_loss, 'soft_depth_loss': soft_depth_loss}

        weight_dict = {'hm_loss': 1, 'hp_loss': 1,
                       'box2d_loss': 1,  'off_loss': 0.5, 'dim_loss': 1,
                       'depth_loss': 1, 'kpd_loss': 0.2,
                       'rot_loss': 1.0, 'soft_depth_loss': 0.2}

        loss = 0
        for key, weight in weight_dict.items():
            if key in loss_stats:
                loss = loss + loss_stats[key] * weight
        loss_stats['total_loss'] = loss
        return loss, loss_stats
