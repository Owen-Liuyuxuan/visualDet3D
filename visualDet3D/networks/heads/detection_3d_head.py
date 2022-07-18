import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.ops import nms
from easydict import EasyDict
import numpy as np
from typing import List, Tuple, Dict


from visualDet3D.networks.heads.losses import SigmoidFocalLoss, ModifiedSmoothL1Loss
from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.networks.utils.utils import calc_iou, BackProjection, BBox3dProjector
from visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt
from visualDet3D.networks.utils.utils import ClipBoxes
from visualDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from visualDet3D.networks.backbones.resnet import BasicBlock
from visualDet3D.networks.lib.ops import ModulatedDeformConvPack
from visualDet3D.networks.lib.look_ground import LookGround

class AnchorBasedDetection3DHead(nn.Module):
    def __init__(self, num_features_in:int=1024,
                       num_classes:int=3,
                       num_regression_loss_terms=12,
                       preprocessed_path:str='',
                       anchors_cfg:EasyDict=EasyDict(),
                       layer_cfg:EasyDict=EasyDict(),
                       loss_cfg:EasyDict=EasyDict(),
                       test_cfg:EasyDict=EasyDict(),
                       read_precompute_anchor:bool=True):
        super(AnchorBasedDetection3DHead, self).__init__()
        self.anchors = Anchors(preprocessed_path=preprocessed_path, readConfigFile=read_precompute_anchor, **anchors_cfg)
        
        self.num_classes = num_classes
        self.num_regression_loss_terms=num_regression_loss_terms
        self.decode_before_loss = getattr(loss_cfg, 'decode_before_loss', False)
        self.loss_cfg = loss_cfg
        self.test_cfg  = test_cfg
        self.build_loss(**loss_cfg)
        self.backprojector = BackProjection()
        self.clipper = ClipBoxes()

        if getattr(layer_cfg, 'num_anchors', None) is None:
            layer_cfg['num_anchors'] = self.anchors.num_anchors
        self.init_layers(**layer_cfg)

    def init_layers(self, num_features_in,
                          num_anchors:int,
                          num_cls_output:int,
                          num_reg_output:int,
                          cls_feature_size:int=1024,
                          reg_feature_size:int=1024,
                          **kwargs):

        self.cls_feature_extraction = nn.Sequential(
            nn.Conv2d(num_features_in, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(cls_feature_size, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),

            nn.Conv2d(cls_feature_size, num_anchors*(num_cls_output), kernel_size=3, padding=1),
            AnchorFlatten(num_cls_output)
        )
        self.cls_feature_extraction[-2].weight.data.fill_(0)
        self.cls_feature_extraction[-2].bias.data.fill_(0)

        self.reg_feature_extraction = nn.Sequential(
            ModulatedDeformConvPack(num_features_in, reg_feature_size, 3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(inplace=True),

            nn.Conv2d(reg_feature_size, num_anchors*num_reg_output, kernel_size=3, padding=1),
            AnchorFlatten(num_reg_output)
        )

        self.reg_feature_extraction[-2].weight.data.fill_(0)
        self.reg_feature_extraction[-2].bias.data.fill_(0)

    def forward(self, inputs):
        cls_preds = self.cls_feature_extraction(inputs['features'])
        reg_preds = self.reg_feature_extraction(inputs['features'])

        return cls_preds, reg_preds
        
    def build_loss(self, focal_loss_gamma=0.0, balance_weight=[0], L1_regression_alpha=9, **kwargs):
        self.focal_loss_gamma = focal_loss_gamma
        self.register_buffer("balance_weights", torch.tensor(balance_weight, dtype=torch.float32))
        self.loss_cls = SigmoidFocalLoss(gamma=focal_loss_gamma, balance_weights=self.balance_weights)
        self.loss_bbox = ModifiedSmoothL1Loss(L1_regression_alpha)

        regression_weight = kwargs.get("regression_weight", [1 for _ in range(self.num_regression_loss_terms)]) #default 12 only use in 3D
        self.register_buffer("regression_weight", torch.tensor(regression_weight, dtype=torch.float))

        self.alpha_loss = nn.BCEWithLogitsLoss(reduction='none')

    def _assign(self, anchor, annotation, 
                    bg_iou_threshold=0.0,
                    fg_iou_threshold=0.5,
                    min_iou_threshold=0.0,
                    match_low_quality=True,
                    gt_max_assign_all=True,
                    **kwargs):
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

    def _encode(self, sampled_anchors, sampled_gt_bboxes, selected_anchors_3d):
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

        targets_dx = (gx - px) / pw
        targets_dy = (gy - py) / ph
        targets_dw = torch.log(gw / pw)
        targets_dh = torch.log(gh / ph)

        targets_cdx = (sampled_gt_bboxes[:, 5] - px) / pw
        targets_cdy = (sampled_gt_bboxes[:, 6] - py) / ph

        targets_cdz = (sampled_gt_bboxes[:, 7] - selected_anchors_3d[:, 0, 0]) / selected_anchors_3d[:, 0, 1]
        targets_cd_sin = (torch.sin(sampled_gt_bboxes[:, 11] * 2) - selected_anchors_3d[:, 1, 0]) / selected_anchors_3d[:, 1, 1]
        targets_cd_cos = (torch.cos(sampled_gt_bboxes[:, 11] * 2) - selected_anchors_3d[:, 2, 0]) / selected_anchors_3d[:, 2, 1]
        targets_w3d = (sampled_gt_bboxes[:, 8]  - selected_anchors_3d[:, 3, 0]) / selected_anchors_3d[:, 3, 1]
        targets_h3d = (sampled_gt_bboxes[:, 9]  - selected_anchors_3d[:, 4, 0]) / selected_anchors_3d[:, 4, 1]
        targets_l3d = (sampled_gt_bboxes[:, 10] - selected_anchors_3d[:, 5, 0]) / selected_anchors_3d[:, 5, 1]

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, 
                         targets_cdx, targets_cdy, targets_cdz,
                         targets_cd_sin, targets_cd_cos,
                         targets_w3d, targets_h3d, targets_l3d), dim=1)

        stds = targets.new([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1])

        targets = targets.div_(stds)

        targets_alpha_cls = (torch.cos(sampled_gt_bboxes[:, 11:12]) > 0).float()
        return targets, targets_alpha_cls #[N, 4]

    def _decode(self, boxes, deltas, anchors_3d_mean_std, label_index, alpha_score):
        std = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1], dtype=torch.float32, device=boxes.device)
        widths  = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x   = boxes[..., 0] + 0.5 * widths
        ctr_y   = boxes[..., 1] + 0.5 * heights

        dx = deltas[..., 0] * std[0]
        dy = deltas[..., 1] * std[1]
        dw = deltas[..., 2] * std[2]
        dh = deltas[..., 3] * std[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        one_hot_mask = torch.nn.functional.one_hot(label_index, anchors_3d_mean_std.shape[1]).bool()
        selected_mean_std = anchors_3d_mean_std[one_hot_mask] #[N]
        mask = selected_mean_std[:, 0, 0] > 0
        
        cdx = deltas[..., 4] * std[4]
        cdy = deltas[..., 5] * std[5]
        pred_cx1 = ctr_x + cdx * widths
        pred_cy1 = ctr_y + cdy * heights
        pred_z   = deltas[...,6] * selected_mean_std[:, 0, 1] + selected_mean_std[:,0, 0]  #[N, 6]
        pred_sin = deltas[...,7] * selected_mean_std[:, 1, 1] + selected_mean_std[:,1, 0] 
        pred_cos = deltas[...,8] * selected_mean_std[:, 2, 1] + selected_mean_std[:,2, 0] 
        pred_alpha = torch.atan2(pred_sin, pred_cos) / 2.0

        pred_w = deltas[...,9]  * selected_mean_std[:, 3, 1] + selected_mean_std[:,3, 0]
        pred_h = deltas[...,10] * selected_mean_std[:,4, 1] + selected_mean_std[:,4, 0]
        pred_l = deltas[...,11] * selected_mean_std[:,5, 1] + selected_mean_std[:,5, 0]

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2,
                                    pred_cx1, pred_cy1, pred_z,
                                    pred_w, pred_h, pred_l, pred_alpha], dim=1)

        pred_boxes[alpha_score[:, 0] < 0.5, -1] += np.pi

        return pred_boxes, mask
        

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

    def _post_process(self, scores, bboxes, labels, P2s):
        
        N = len(scores)
        bbox2d = bboxes[:, 0:4]
        bbox3d = bboxes[:, 4:] #[cx, cy, z, w, h, l, alpha]

        bbox3d_state_3d = self.backprojector.forward(bbox3d, P2s[0]) #[x, y, z, w, h, l, alpha]
        for i in range(N):
            if bbox3d_state_3d[i, 2] > 3 and labels[i] == 0:
                bbox3d[i] = post_opt(
                    bbox2d[i], bbox3d_state_3d[i], P2s[0].cpu().numpy(),
                    bbox3d[i, 0].item(), bbox3d[i, 1].item()
                )
        bboxes = torch.cat([bbox2d, bbox3d], dim=-1)
        return scores, bboxes, labels

    def get_anchor(self, img_batch, P2):
        is_filtering = getattr(self.loss_cfg, 'filter_anchor', True)
        if not self.training:
            is_filtering = getattr(self.test_cfg, 'filter_anchor', is_filtering)

        anchors, useful_mask, anchor_mean_std = self.anchors(img_batch, P2, is_filtering=is_filtering)
        return_dict=dict(
            anchors=anchors, #[1, N, 4]
            mask=useful_mask, #[B, N]
            anchor_mean_std_3d = anchor_mean_std  #[N, C, K=6, 2]
        )
        return return_dict

    def _get_anchor_3d(self, anchors, anchor_mean_std_3d, assigned_labels):
        """
            anchors: [N_pos, 4] only positive anchors
            anchor_mean_std_3d: [N_pos, C, K=6, 2]
            assigned_labels: torch.Longtensor [N_pos, ]

            return:
                selected_mask = torch.bool [N_pos, ]
                selected_anchors_3d:  [N_selected, K, 2]
        """
        one_hot_mask = torch.nn.functional.one_hot(assigned_labels, self.num_classes).bool()
        selected_anchor_3d = anchor_mean_std_3d[one_hot_mask]

        selected_mask = selected_anchor_3d[:, 0, 0] > 0 #only z > 0, filter out anchors with good variance and mean
        selected_anchor_3d = selected_anchor_3d[selected_mask]

        return selected_mask, selected_anchor_3d

    def get_bboxes(self, cls_scores, reg_preds, anchors, P2s, img_batch=None):
        
        assert cls_scores.shape[0] == 1 # batch == 1
        cls_scores = cls_scores.sigmoid()

        cls_score = cls_scores[0][..., 0:self.num_classes]
        alpha_score = cls_scores[0][..., self.num_classes:self.num_classes+1]
        reg_pred  = reg_preds[0]
        
        anchor = anchors['anchors'][0] #[N, 4]
        anchor_mean_std_3d = anchors['anchor_mean_std_3d'] #[N, K, 2]
        useful_mask = anchors['mask'][0] #[N, ]

        anchor = anchor[useful_mask]
        cls_score = cls_score[useful_mask]
        alpha_score = alpha_score[useful_mask]
        reg_pred = reg_pred[useful_mask]
        anchor_mean_std_3d = anchor_mean_std_3d[useful_mask] #[N, K, 2]

        score_thr = getattr(self.test_cfg, 'score_thr', 0.5)
        max_score, label = cls_score.max(dim=-1) 

        high_score_mask = (max_score > score_thr)

        anchor      = anchor[high_score_mask, :]
        anchor_mean_std_3d = anchor_mean_std_3d[high_score_mask, :]
        cls_score   = cls_score[high_score_mask, :]
        alpha_score = alpha_score[high_score_mask, :]
        reg_pred    = reg_pred[high_score_mask, :]
        max_score   = max_score[high_score_mask]
        label       = label[high_score_mask]


        bboxes, mask = self._decode(anchor, reg_pred, anchor_mean_std_3d, label, alpha_score)
        if img_batch is not None:
            bboxes = self.clipper(bboxes, img_batch)
        cls_score = cls_score[mask]
        max_score = max_score[mask]
        bboxes    = bboxes[mask]

        cls_agnostic = getattr(self.test_cfg, 'cls_agnositc', True) # True -> directly NMS; False -> NMS with offsets different categories will not collide
        nms_iou_thr  = getattr(self.test_cfg, 'nms_iou_thr', 0.5)
        

        if cls_agnostic:
            keep_inds = nms(bboxes[:, :4], max_score, nms_iou_thr)
        else:
            max_coordinate = bboxes.max()
            nms_bbox = bboxes[:, :4] + label.float().unsqueeze() * (max_coordinate)
            keep_inds = nms(nms_bbox, max_score, nms_iou_thr)

        bboxes      = bboxes[keep_inds]
        max_score   = max_score[keep_inds]
        label       = label[keep_inds]

        is_post_opt = getattr(self.test_cfg, 'post_optimization', False)
        if is_post_opt:
            max_score, bboxes, label = self._post_process(max_score, bboxes, label, P2s)

        return max_score, bboxes, label

    def loss(self, cls_scores, reg_preds, anchors, annotations, P2s):
        batch_size = cls_scores.shape[0]

        anchor = anchors['anchors'][0] #[N, 4]
        anchor_mean_std_3d = anchors['anchor_mean_std_3d']
        cls_loss = []
        reg_loss = []
        number_of_positives = []
        for j in range(batch_size):
            
            reg_pred  = reg_preds[j]
            cls_score = cls_scores[j][..., 0:self.num_classes]
            alpha_score = cls_scores[j][..., self.num_classes:self.num_classes+1]

            # selected by mask
            useful_mask = anchors['mask'][j] #[N]
            anchor_j = anchor[useful_mask]
            anchor_mean_std_3d_j = anchor_mean_std_3d[useful_mask]
            reg_pred = reg_pred[useful_mask]
            cls_score = cls_score[useful_mask]
            alpha_score = alpha_score[useful_mask]

            # only select useful bbox_annotations
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]#[k]

            if len(bbox_annotation) == 0:
                cls_loss.append(torch.tensor(0).cuda().float())
                reg_loss.append(reg_preds.new_zeros(self.num_regression_loss_terms))
                number_of_positives.append(0)
                continue

            assignement_result_dict = self._assign(anchor_j, bbox_annotation, **self.loss_cfg)
            sampling_result_dict    = self._sample(assignement_result_dict, anchor_j, bbox_annotation)
        
            num_valid_anchors = anchor_j.shape[0]
            labels = anchor_j.new_full((num_valid_anchors, self.num_classes),
                                    -1, # -1 not computed, binary for each class
                                    dtype=torch.float)

            pos_inds = sampling_result_dict['pos_inds']
            neg_inds = sampling_result_dict['neg_inds']
            
            if len(pos_inds) > 0:
                pos_assigned_gt_label = bbox_annotation[sampling_result_dict['pos_assigned_gt_inds'], 4].long()
                
                selected_mask, selected_anchor_3d = self._get_anchor_3d(
                    sampling_result_dict['pos_bboxes'],
                    anchor_mean_std_3d_j[pos_inds],
                    pos_assigned_gt_label,
                )
                if len(selected_anchor_3d) > 0:
                    pos_inds = pos_inds[selected_mask]
                    pos_bboxes    = sampling_result_dict['pos_bboxes'][selected_mask]
                    pos_gt_bboxes = sampling_result_dict['pos_gt_bboxes'][selected_mask]
                    pos_assigned_gt = sampling_result_dict['pos_assigned_gt_inds'][selected_mask]

                    pos_bbox_targets, targets_alpha_cls = self._encode(
                        pos_bboxes, pos_gt_bboxes, selected_anchor_3d
                    ) #[N, 12], [N, 1]
                    label_index = pos_assigned_gt_label[selected_mask]
                    labels[pos_inds, :] = 0
                    labels[pos_inds, label_index] = 1

                    pos_anchor = anchor[pos_inds]
                    pos_alpha_score = alpha_score[pos_inds]
                    if self.decode_before_loss:
                        pos_prediction_decoded, mask = self._decode(pos_anchor, reg_pred[pos_inds],  anchor_mean_std_3d_j[pos_inds], label_index, pos_alpha_score)
                        pos_target_decoded, _     = self._decode(pos_anchor, pos_bbox_targets,  anchor_mean_std_3d_j[pos_inds], label_index, pos_alpha_score)
                        reg_loss_j = self.loss_bbox(pos_prediction_decoded[mask], pos_target_decoded[mask])
                        alpha_loss_j = self.alpha_loss(pos_alpha_score[mask], targets_alpha_cls[mask])
                        loss_j = torch.cat([reg_loss_j, alpha_loss_j], dim=1) * self.regression_weight #[N, 12]
                        reg_loss.append(loss_j.mean(dim=0)) #[13]
                        number_of_positives.append(bbox_annotation.shape[0])
                    else:
                        reg_loss_j = self.loss_bbox(pos_bbox_targets, reg_pred[pos_inds]) 
                        alpha_loss_j = self.alpha_loss(pos_alpha_score, targets_alpha_cls)
                        loss_j = torch.cat([reg_loss_j, alpha_loss_j], dim=1) * self.regression_weight #[N, 13]
                        reg_loss.append(loss_j.mean(dim=0)) #[13]
                        number_of_positives.append(bbox_annotation.shape[0])
            else:
                reg_loss.append(reg_preds.new_zeros(self.num_regression_loss_terms))
                number_of_positives.append(bbox_annotation.shape[0])

            if len(neg_inds) > 0:
                labels[neg_inds, :] = 0
            
            cls_loss.append(self.loss_cls(cls_score, labels).sum() / (len(pos_inds) + len(neg_inds)))
        
        weights = reg_pred.new(number_of_positives).unsqueeze(1) #[B, 1]
        cls_loss = torch.stack(cls_loss).mean(dim=0, keepdim=True)
        reg_loss = torch.stack(reg_loss, dim=0) #[B, 12]

        weighted_regression_losses = torch.sum(weights * reg_loss / (torch.sum(weights) + 1e-6), dim=0)
        reg_loss = weighted_regression_losses.mean(dim=0, keepdim=True)

        return cls_loss, reg_loss, dict(cls_loss=cls_loss, reg_loss=reg_loss, total_loss=cls_loss + reg_loss)

class StereoHead(AnchorBasedDetection3DHead):
    def init_layers(self, num_features_in,
                          num_anchors:int,
                          num_cls_output:int,
                          num_reg_output:int,
                          cls_feature_size:int=1024,
                          reg_feature_size:int=1024,
                          **kwargs):

        self.cls_feature_extraction = nn.Sequential(
            nn.Conv2d(num_features_in, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(cls_feature_size, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),

            nn.Conv2d(cls_feature_size, num_anchors*(num_cls_output), kernel_size=3, padding=1),
            AnchorFlatten(num_cls_output)
        )
        self.cls_feature_extraction[-2].weight.data.fill_(0)
        self.cls_feature_extraction[-2].bias.data.fill_(0)

        self.reg_feature_extraction = nn.Sequential(

            ConvBnReLU(num_features_in, reg_feature_size, (3, 3)),
            BasicBlock(reg_feature_size, reg_feature_size),
            nn.ReLU(),
            nn.Conv2d(reg_feature_size, num_anchors*num_reg_output, kernel_size=3, padding=1),
            AnchorFlatten(num_reg_output)
        )

        self.reg_feature_extraction[-2].weight.data.fill_(0)
        self.reg_feature_extraction[-2].bias.data.fill_(0)
