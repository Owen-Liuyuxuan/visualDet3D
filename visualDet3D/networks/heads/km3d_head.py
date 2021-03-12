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
from visualDet3D.networks.lib.blocks import AnchorFlatten
from visualDet3D.networks.lib.ops import ModulatedDeformConvPack
from visualDet3D.networks.lib.look_ground import LookGround
from visualDet3D.networks.utils.rtm3d_utils import _transpose_and_gather_feat, compute_rot_loss, gen_position, Position_loss, _nms, _topk_channel, _topk
from visualDet3D.utils.utils import convertRot2Alpha

class KM3DHead(nn.Module):
    """Some Information about KM3DHead"""
    def __init__(self, num_classes:int=3,
                       num_joints:int=9,
                       max_objects:int=32,
                       layer_cfg=EasyDict(),
                       loss_cfg=EasyDict(),
                       test_cfg=EasyDict()):
        super(KM3DHead, self).__init__()
        self._init_layers(**layer_cfg)
        self.build_loss(**loss_cfg)
        self.test_cfg = test_cfg
        const = torch.Tensor(
        [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
        [-1, 0], [0, -1], [-1, 0], [0, -1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('const', const) # self.const

        self.num_classes = num_classes
        self.num_joints  = num_joints
        self.max_objects = max_objects
        self.clipper = ClipBoxes()

    def build_loss(self, 
                   gamma=2.0,
                   output_w = 1280,
                   rampup_length = 100,
                   **kwargs):
        pass #self.cls_hm_loss = SigmoidFocalLoss(gamma=gamma)
        self.position_loss = Position_loss(output_w=output_w)
        self.rampup_length = rampup_length

    def exp_rampup(self, epoch=0):
        if epoch < self.rampup_length:
            epoch = np.clip(epoch, 0.0, self.rampup_length)
            phase = 1.0 - epoch / self.rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    
    @staticmethod
    def _neg_loss(pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred_prob = torch.sigmoid(pred)

        pos_loss = nn.functional.logsigmoid(pred) * torch.pow(1 - pred_prob, 2) * pos_inds
        pos_loss = torch.where(
            pred_prob > 0.99,
            torch.zeros_like(pos_loss),
            pos_loss
        )
        neg_loss = nn.functional.logsigmoid(- pred) * torch.pow(pred_prob, 2) * neg_weights * neg_inds
        neg_loss = torch.where(
            pred_prob < 0.01,
            torch.zeros_like(neg_loss),
            neg_loss
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    @staticmethod
    def _RegWeightedL1Loss(output, mask, ind, target, dep):
        dep=dep.squeeze(2)
        dep[dep<5]=dep[dep<5]*0.01
        dep[dep >= 5] = torch.log10(dep[dep >=5]-4)+0.1
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        #losss=torch.abs(pred * mask-target * mask)
        #loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss=torch.abs(pred * mask-target * mask)
        loss=torch.sum(loss,dim=2)*dep
        loss=loss.sum()
        loss = loss / (mask.sum() + 1e-4)

        return loss

    @staticmethod
    def _RegL1Loss(output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

    @staticmethod
    def _RotLoss(output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss

    def _init_layers(self, 
                    input_features=256,
                    head_features=64,
                    head_dict=dict(),
                     **kwargs):
        # self.head_dict = head_dict 
        self.head_layers = nn.ModuleDict()
        for head_name, num_output in head_dict.items():
            self.head_layers[head_name] = nn.Sequential(
                    nn.Conv2d(input_features, head_features, 3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_features, num_output, 1)
                )
            
            if 'hm' in head_name:
                output_layer = self.head_layers[head_name][-1]
                nn.init.constant_(output_layer.bias, -2.19)

            else:
                output_layer = self.head_layers[head_name][-1]
                nn.init.normal_(output_layer.weight, std=0.001)
                nn.init.constant_(output_layer.bias, 0)

    def _decode(self, heat, wh, kps,dim,rot, prob=None,reg=None, hm_hp=None, hp_offset=None, K=100,meta=None,const=None):

        batch, cat, height, width = heat.size()
        num_joints = kps.shape[1] // 2
        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        # hm_show,_=torch.max(hm_hp,1)
        # hm_show=hm_show.squeeze(0)
        # hm_show=hm_show.detach().cpu().numpy().copy()
        # plt.imshow(hm_show, 'gray')
        # plt.show()

        heat = _nms(heat)
        scores, inds, clses, ys, xs = _topk(heat, K=K)

        kps = _transpose_and_gather_feat(kps, inds)
        kps = kps.view(batch, K, num_joints * 2)
        kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
        kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
        if reg is not None:
            reg = _transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        dim = _transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, K, 3)
        # dim[:, :, 0] = torch.exp(dim[:, :, 0]) * 1.63
        # dim[:, :, 1] = torch.exp(dim[:, :, 1]) * 1.53
        # dim[:, :, 2] = torch.exp(dim[:, :, 2]) * 3.88
        rot = _transpose_and_gather_feat(rot, inds)
        rot = rot.view(batch, K, 8)
        prob = _transpose_and_gather_feat(prob, inds)[:,:,0]
        prob = prob.view(batch, K, 1)
        if hm_hp is not None:
            hm_hp = _nms(hm_hp)
            thresh = 0.1
            kps = kps.view(batch, K, num_joints, 2).permute(
                0, 2, 1, 3).contiguous()  # b x J x K x 2
            reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
            hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
            if hp_offset is not None:
                hp_offset = _transpose_and_gather_feat(
                    hp_offset, hm_inds.view(batch, -1))
                hp_offset = hp_offset.view(batch, num_joints, K, 2)
                hm_xs = hm_xs + hp_offset[:, :, :, 0]
                hm_ys = hm_ys + hp_offset[:, :, :, 1]
            else:
                hm_xs = hm_xs + 0.5
                hm_ys = hm_ys + 0.5
            mask = (hm_score > thresh).float()
            hm_score = (1 - mask) * -1 + mask * hm_score
            hm_ys = (1 - mask) * (-10000) + mask * hm_ys
            hm_xs = (1 - mask) * (-10000) + mask * hm_xs
            hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
                2).expand(batch, num_joints, K, K, 2)
            dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
            min_dist, min_ind = dist.min(dim=3)  # b x J x K
            hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
            min_dist = min_dist.unsqueeze(-1)
            min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
                batch, num_joints, K, 1, 2)
            hm_kps = hm_kps.gather(3, min_ind)
            hm_kps = hm_kps.view(batch, num_joints, K, 2)
            l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
                (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
            mask = (mask > 0).float().expand(batch, num_joints, K, 2)
            kps = (1 - mask) * hm_kps + mask * kps
            kps = kps.permute(0, 2, 1, 3).contiguous().view(
                batch, K, num_joints * 2)
            hm_score=hm_score.permute(0, 2, 1, 3).squeeze(3).contiguous()
        else:
            hm_score = kps.new_zeros([1, K, 9])# dets[mask, 26:35]

        kps *= 4 # restore back to scale 1
        bboxes *= 4 # restore back to scale 1
        
        position,rot_y, alpha, kps_inv=gen_position(kps,dim,rot,meta,const)

        detections = torch.cat([bboxes, scores, kps_inv, dim,hm_score,rot_y, position,prob,clses, alpha], dim=2)

        return detections


    def get_bboxes(self, output:dict, P2, img_batch=None):
        output['hm'] = torch.sigmoid(output['hm'])
        output['hm_hp'] = torch.sigmoid(output['hm_hp'])
        reg = output['reg']
        hm_hp = output['hm_hp']
        hp_offset = output['hp_offset']
        dets = self._decode(
            output['hm'], output['wh'], output['hps'], output['dim'], output['rot'], prob=output['prob'], reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=100, const=self.const, meta=dict(calib=P2)
        )[0]

        score_threshold = getattr(self.test_cfg, 'score_thr', 0.1)
        mask = dets[:, 4] > score_threshold#[K]
        bbox2d = dets[mask, 0:4]
        scores = dets[mask, 4:5] #[K, 1]
        kps_inv = dets[mask, 5:23] #[K, 18]
        dims   = dets[mask, 23:26] #[w, h, l] ? 
        hm_score = dets[mask, 26:35]
        rot_y = dets[mask, 35:36]
        position = dets[mask, 36:39]
        prob = dets[mask, 39:40]
        cls_indexes = dets[mask, 40:41].long()
        alpha = dets[mask, 41:42]
        
        ## Project back to camera frame for final output
        p2 = P2[0] #[3, 4]
        fx = p2[0, 0]
        fy = p2[1, 1]
        cx = p2[0, 2]
        cy = p2[1, 2]
        tx = p2[0, 3]
        ty = p2[1, 3]
        z3d = position[:, 2:3] #[N, 1]
        cx3d = (position[:, 0:1] * fx + tx + cx * z3d) / z3d
        cy3d = (position[:, 1:2] * fy + ty + cy * z3d) / z3d
    
        if img_batch is not None:
            bbox2d = self.clipper(bbox2d, img_batch)

        bbox3d_3d = torch.cat(
            [bbox2d,  cx3d, cy3d, z3d, dims, alpha], dim=1                  #cx, cy, z, w, h, l, alpha
        )
        
        
        cls_agnostic = getattr(self.test_cfg, 'cls_agnositc', True) # True -> directly NMS; False -> NMS with offsets, different categories will not collide
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

        #output['hm'] = torch.sigmoid(output['hm'])
        #output['hm_hp'] = torch.sigmoid(output['hm_hp'])

        hm_loss = self._neg_loss(output['hm'], annotations['hm'])
        hp_loss = self._RegWeightedL1Loss(output['hps'],annotations['hps_mask'], annotations['ind'], annotations['hps'],annotations['dep'])

        wh_loss = self._RegL1Loss(output['wh'], annotations['reg_mask'],annotations['ind'], annotations['wh'])
        dim_loss = self._RegL1Loss(output['dim'], annotations['reg_mask'],annotations['ind'], annotations['dim'])

        rot_loss = self._RotLoss(output['rot'], annotations['reg_mask'], annotations['ind'], annotations['rotbin'], annotations['rotres'])
        off_loss = self._RegL1Loss(output['reg'], annotations['reg_mask'], annotations['ind'], annotations['reg'])

        hp_offset_loss = self._RegL1Loss(output['hp_offset'], annotations['hp_mask'], annotations['hp_ind'], annotations['hp_offset'])
        hm_hp_loss = self._neg_loss(output['hm_hp'], annotations['hm_hp'])
        coor_loss, prob_loss, box_score = self.position_loss(output, annotations, P2)

        loss_stats = {'loss': box_score, 'hm_loss': hm_loss, 'hp_loss': hp_loss,
                      'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss,'dim_loss': dim_loss,
                      'rot_loss':rot_loss,'prob_loss':prob_loss,'box_score':box_score,'coor_loss':coor_loss}

        weight_dict = {'hm_loss': 1, 'hp_loss': 1,
                       'hm_hp_loss': 1, 'hp_offset_loss': 1,
                       'wh_loss': 0.1, 'off_loss': 1, 'dim_loss': 2,
                       'rot_loss': 0.2, 'prob_loss': self.exp_rampup(epoch), 'coor_loss': self.exp_rampup(epoch)}

        loss = 0
        for key, weight in weight_dict.items():
            if key in loss_stats:
                loss = loss + loss_stats[key] * weight
        loss_stats['total_loss'] = loss
        return loss, loss_stats

    def forward(self, x):
        ret = {}
        for head in self.head_layers:
            ret[head] = self.head_layers[head](x)
        return ret
