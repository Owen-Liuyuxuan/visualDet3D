import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.ops import nms
import numpy as np
from functools import partial
from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.networks.utils.utils import calc_iou
from visualDet3D.networks.lib.blocks import ConvReLU, AnchorFlatten
from visualDet3D.networks.heads.losses import SigmoidFocalLoss, IoULoss

class RetinanetHead(nn.Module):
    """Some Information about RetinanetHead"""
    def __init__(self,  stacked_convs=4,
                        in_channels=256,
                        feat_channels=256,
                        num_classes=3,
                        reg_output =4,
                        target_stds =[1.0, 1.0, 1.0, 1.0],
                        target_means=[ .0,  .0,  .0,  .0],
                        anchors_cfg=dict(),
                        loss_cfg = dict(),
                        test_cfg = dict()
                        ):
        super(RetinanetHead, self).__init__()
        self.anchors = Anchors(preprocessed_path=None, readConfigFile=False, **anchors_cfg)
        
        self.stacked_convs = stacked_convs
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_clasess = num_classes
        self.target_stds = target_stds
        self.target_means = target_means
        self.loss_cfg     = loss_cfg
        self.test_cfg     = test_cfg

        ## Construct Convolutions
        if stacked_convs > 0:
            in_channel_list = [in_channels] + [feat_channels for _ in range(stacked_convs - 1)]
        else:
            in_channel_list = []
        self.cls_conv = nn.Sequential(
            *[
                ConvReLU(in_channel_list[i], feat_channels, (3, 3) )  # no norm by default
                for i in range(len(in_channel_list))  
            ]
        )
        self.reg_conv = nn.Sequential(
            *[
                ConvReLU(in_channel_list[i], feat_channels, (3, 3) ) 
                for i in range(len(in_channel_list))  # no norm by default
            ]
        )
        self.retina_cls = nn.Sequential(
            nn.Conv2d(feat_channels, self.anchors.num_anchor_per_scale * num_classes, 3, padding=1),
            AnchorFlatten(num_classes)
        ) # shared head
        self.retina_reg = nn.Sequential(
            nn.Conv2d(feat_channels, self.anchors.num_anchor_per_scale * reg_output, 3, padding=1),
            AnchorFlatten(reg_output)
        )

        cls_prior = 0.01
        self.retina_cls[0].weight.data.fill_(0)
        self.retina_cls[0].bias.data.fill_(np.log((cls_prior) / (1-cls_prior) )) # prior from retinanet/pytorch

        self.retina_reg[0].weight.data.fill_(0) # prior from retinanet/pytorch
        self.retina_reg[0].bias.data.fill_(0)

        self.build_loss(**loss_cfg)

    def build_loss(self, gamma=0.0, balance_weights=0, **kwargs):
        self.gamma = gamma
        self.register_buffer("balance_weights", torch.tensor(balance_weights, dtype=torch.float32))
        self.loss_cls = SigmoidFocalLoss(gamma=gamma, balance_weights=self.balance_weights)
        self.loss_bbox = IoULoss() #nn.L1Loss(reduction='none')

    def forward(self, feats):

        cls_scores = []
        reg_preds  = []
        for feat in feats:
            cls_feat = self.cls_conv(feat)
            reg_feat = self.reg_conv(feat)

            cls_scores.append(self.retina_cls(cls_feat))
            reg_preds.append(self.retina_reg(reg_feat))
        
        cls_scores = torch.cat(cls_scores, dim=1) # [B, N, num_class]
        reg_preds  = torch.cat(reg_preds,  dim=1) # [B, N, 4]

        return (cls_scores, reg_preds)


    def get_anchor(self, img_batch):
        return self.anchors(img_batch)

    def _assign(self, anchor, annotation, 
                    bg_iou_threshold=0.0,
                    fg_iou_threshold=0.5,
                    min_iou_threshold=0.0,
                    match_low_quality=True,
                    gt_max_assign_all=True, **kwargs):
        """
            anchor: [N, 4]
            annotation: [num_gt, 4]:
        """
        N = anchor.shape[0]
        num_gt = annotation.shape[0]
        assigned_gt_inds = anchor.new_full(
            (N, ),
            -1, dtype=torch.long
        ) #[N, ] torch.long
        max_overlaps = anchor.new_zeros((N, ))
        assigned_labels = anchor.new_full((N, ),
            -1,
            dtype=torch.long)

        if num_gt == 0:
            assigned_gt_inds = anchor.new_full(
                (N, ),
                0, dtype=torch.long
            ) #[N, ] torch.long
            return_dict = dict(
                num_gt=num_gt,
                assigned_gt_inds = assigned_gt_inds,
                max_overlaps = max_overlaps,
                labels=assigned_labels
            )
            return return_dict

        IoU = calc_iou(anchor, annotation[:, :4]) # num_anchors x num_annotations

        # max for anchor
        max_overlaps, argmax_overlaps = IoU.max(dim=1) # num_anchors

        # max for gt
        gt_max_overlaps, gt_argmax_overlaps = IoU.max(dim=0) #num_gt

        # assign negative
        assigned_gt_inds[(max_overlaps >=0) & (max_overlaps < bg_iou_threshold)] = 0

        # assign positive
        pos_inds = max_overlaps >= fg_iou_threshold
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if match_low_quality:
            for i in range(num_gt):
                if gt_max_overlaps[i] >= min_iou_threshold:
                    if gt_max_assign_all:
                        max_iou_inds = IoU[:, i] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i+1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i+1

        
        assigned_labels = assigned_gt_inds.new_full((N, ), -1)
        pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False
            ).squeeze()
        if pos_inds.numel()>0:
            assigned_labels[pos_inds] = annotation[assigned_gt_inds[pos_inds] - 1, 4].long()

        return_dict = dict(
            num_gt = num_gt,
            assigned_gt_inds = assigned_gt_inds,
            max_overlaps  = max_overlaps,
            labels = assigned_labels
        )
        return return_dict

    def _sample(self, assignment_result, anchors, gt_bboxes):
        """
            Pseudo sampling
        """
        pos_inds = torch.nonzero(
                assignment_result['assigned_gt_inds'] > 0, as_tuple=False
            ).unsqueeze(-1).unique()
        neg_inds = torch.nonzero(
                assignment_result['assigned_gt_inds'] == 0, as_tuple=False
            ).unsqueeze(-1).unique()
        gt_flags = anchors.new_zeros(anchors.shape[0], dtype=torch.uint8) #

        pos_assigned_gt_inds = assignment_result['assigned_gt_inds'] - 1

        if gt_bboxes.numel() == 0:
            pos_gt_bboxes = gt_bboxes.new_zeros([0, 4])
        else:
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds[pos_inds], :]
        return_dict = dict(
            pos_inds = pos_inds,
            neg_inds = neg_inds,
            pos_bboxes = anchors[pos_inds],
            neg_bboxes = anchors[neg_inds],
            pos_gt_bboxes = pos_gt_bboxes,
            pos_assigned_gt_inds = pos_assigned_gt_inds[pos_inds],
        )
        return return_dict

    def _encode(self, sampled_anchors, sampled_gt_bboxes):
        assert sampled_anchors.shape[0] == sampled_gt_bboxes.shape[0]

        sampled_anchors = sampled_anchors.float()
        sampled_gt_bboxes = sampled_gt_bboxes.float()
        px = (sampled_anchors[..., 0] + sampled_anchors[..., 2]) * 0.5
        py = (sampled_anchors[..., 1] + sampled_anchors[..., 3]) * 0.5
        pw = sampled_anchors[..., 2] - sampled_anchors[..., 0]
        ph = sampled_anchors[..., 3] - sampled_anchors[..., 1]

        gx = (sampled_gt_bboxes[..., 0] + sampled_gt_bboxes[..., 2]) * 0.5
        gy = (sampled_gt_bboxes[..., 1] + sampled_gt_bboxes[..., 3]) * 0.5
        gw = sampled_gt_bboxes[..., 2] - sampled_gt_bboxes[..., 0]
        gh = sampled_gt_bboxes[..., 3] - sampled_gt_bboxes[..., 1]

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        deltas = torch.stack([dx, dy, dw, dh], dim=-1)

        means = deltas.new_tensor(self.target_means).unsqueeze(0)
        stds = deltas.new_tensor(self.target_stds).unsqueeze(0)
        deltas = deltas.sub_(means).div_(stds)
        return deltas #[N, 4]

    def _decode(self, anchors, pred_bboxes):
        means = pred_bboxes.new_tensor(self.target_means).unsqueeze(0)
        stds = pred_bboxes.new_tensor(self.target_stds).unsqueeze(0)

        denorm_deltas = pred_bboxes * stds + means
        dx = denorm_deltas[:, 0::4]
        dy = denorm_deltas[:, 1::4]
        dw = denorm_deltas[:, 2::4]
        dh = denorm_deltas[:, 3::4]

        # Compute center of each roi
        px = ((anchors[:, 0] + anchors[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
        py = ((anchors[:, 1] + anchors[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
        # Compute width/height of each roi
        pw = (anchors[:, 2] - anchors[:, 0]).unsqueeze(1).expand_as(dw)
        ph = (anchors[:, 3] - anchors[:, 1]).unsqueeze(1).expand_as(dh)
        # Use exp(network energy) to enlarge/shrink each roi
        gw = pw * dw.exp()
        gh = ph * dh.exp()
        # Use network energy to shift the center of each roi
        gx = px + pw * dx
        gy = py + ph * dy
        # Convert center-xy/width/height to top-left, bottom-right
        x1 = gx - gw * 0.5
        y1 = gy - gh * 0.5
        x2 = gx + gw * 0.5
        y2 = gy + gh * 0.5
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
        return bboxes #[N, 4]
    
    def get_bboxes(self, cls_scores, reg_preds, anchors):

        assert cls_scores.shape[0] == 1 # batch == 1
        cls_scores = cls_scores.sigmoid()

        cls_score = cls_scores[0]
        reg_pred  = reg_preds[0]
        
        anchor    = anchors[0]

        pre_nms_num = getattr(self.test_cfg, 'nms_pre', 1000)
        if pre_nms_num > 0 and cls_scores.shape[1] > pre_nms_num:
            max_score, _ = cls_score.max(dim=-1) # get foregound cls

            _, topkinds = max_score.topk(pre_nms_num)
            anchor      = anchor[topkinds, :]
            cls_score   = cls_score[topkinds, :]
            reg_pred    = reg_pred[topkinds, :]
            max_score   = max_score[topkinds]

        bboxes = self._decode(anchor, reg_pred)

        cls_agnostic = getattr(self.test_cfg, 'cls_agnositc', True) # True -> directly NMS; False -> NMS with offsets different categories will not collide
        nms_iou_thr  = getattr(self.test_cfg, 'nms_iou_thr', 0.5)
        
        max_score, label = cls_score.max(dim=-1)

        if cls_agnostic:
            keep_inds = nms(bboxes, max_score, nms_iou_thr)
        else:
            max_coordinate = bboxes.max()
            nms_bbox = bboxes + label.float().unsqueeze() * (max_coordinate)
            keep_inds = nms(nms_bbox, max_score, nms_iou_thr)


        bboxes      = bboxes[keep_inds]
        max_score   = max_score[keep_inds]
        label       = label[keep_inds]

        score_thr = getattr(self.test_cfg, 'score_thr', 0.5)
        keep_inds_high = []
        for i in range(len(bboxes)):
            if max_score[i] > score_thr:
                keep_inds_high.append(i)
        keep_inds_high = keep_inds.new(keep_inds_high)

        bboxes      = bboxes[keep_inds_high]
        max_score   = max_score[keep_inds_high]
        label       = label[keep_inds_high]

        return max_score, bboxes, label

    def loss(self, cls_scores, reg_preds, anchors, annotations):
        batch_size = cls_scores.shape[0]

        anchor = anchors[0] #[N, 4]
        cls_loss = 0
        reg_loss = 0
        number_of_positives = 1e-4
        for j in range(batch_size):
            
            reg_pred  = reg_preds[j]
            cls_score = cls_scores[j]

            # only select useful bbox_annotations
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]#[k]


            assignement_result_dict = self._assign(anchor, bbox_annotation, **self.loss_cfg)
            sampling_result_dict    = self._sample(assignement_result_dict, anchor, bbox_annotation)
        
            num_valid_anchors = anchor.shape[0]
            bbox_targets = torch.zeros_like(anchor)
            bbox_weights = torch.zeros_like(anchor)
            labels = anchors.new_full((num_valid_anchors, self.num_clasess),
                                    -1, # -1 not computed, binary for each class
                                    dtype=torch.float)

            pos_inds = sampling_result_dict['pos_inds']
            neg_inds = sampling_result_dict['neg_inds']
            number_of_positives += len(pos_inds)
            if len(pos_inds) > 0:
                pos_bbox_targets = self._encode(
                    sampling_result_dict['pos_bboxes'], sampling_result_dict['pos_gt_bboxes']
                )
                bbox_targets[pos_inds, :] = pos_bbox_targets
                bbox_weights[pos_inds, :] = 1.0
                labels[pos_inds, :] = 0
                labels[pos_inds, bbox_annotation[sampling_result_dict['pos_assigned_gt_inds'], 4].long()] = 1

                pos_anchor = anchor[pos_inds]
                pos_prediction_decoded = self._decode(pos_anchor, reg_pred[pos_inds])
                pos_target_decoded     = self._decode(pos_anchor, pos_bbox_targets)

                reg_loss += self.loss_bbox(pos_prediction_decoded, pos_target_decoded).sum()
            
            if len(neg_inds) > 0:
                labels[neg_inds, :] = 0

            
            cls_loss += self.loss_cls(cls_score, labels).sum()#(len(pos_inds) + len(neg_inds))

        cls_loss /= float(number_of_positives)
        reg_loss /= float(number_of_positives)
        return cls_loss, reg_loss, dict(cls_loss=cls_loss, reg_loss=reg_loss, total_loss=cls_loss + reg_loss)
