# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from ..predictor import pred_utils
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d


@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None,
                 # for prediction
                 do_pred=False,
                 pred_embed_dims=256,
                 relative_pred=False,
                 agents_layer_0=False,
                 agents_layer_0_num=2,
                 add_branch=False,
                 predictor=None,
                 collect_keys_pred=['pred_mapping', 'pred_polyline_spans', 'pred_matrix', 'future_traj', 'future_traj_is_valid', 'past_traj', 'past_traj_is_valid'],
                 ):
        super(Petr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only

        self.do_pred = do_pred
        self.collect_keys_pred = collect_keys_pred
        self.pred_embed_dims = pred_embed_dims
        self.relative_pred = relative_pred
        self.agents_layer_0 = agents_layer_0
        self.add_branch = add_branch
        if self.do_pred:
            from ..predictor.predictor_vectornet import VectorNet
            self.predictor = VectorNet(**predictor)
            self.empty_linear = nn.Linear(pred_embed_dims, pred_embed_dims)

            if self.agents_layer_0:
                from ..predictor import predictor_lib
                self.agents_layer_mlp_0 = nn.Sequential(*[predictor_lib.MLP(pred_embed_dims, pred_embed_dims) for _ in range(agents_layer_0_num)])
            
            if self.add_branch:
                from mmcv.utils import build_from_cfg
                from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER)
                self.add_branch_mlp = predictor_lib.MLP(pred_embed_dims, pred_embed_dims)
                self.add_branch_attention = build_from_cfg(pred_utils.get_attention_cfg(), TRANSFORMER_LAYER)


    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN/B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/len_queue), C, H, W)


        return img_feats_reshaped


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T, training_mode)
        return img_feats


    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_bboxes=None,
                            gt_labels=None,
                            img_metas=None,
                            centers2d=None,
                            depths=None,
                            gt_bboxes_ignore=None,
                            **data):
        losses = dict()
        B = data['img'].size(0)
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                if key not in self.collect_keys_pred:    # for pred.
                    data_t[key] = data[key][:, i]
                else:
                    data_t[key] = [data[key][b][i] for b in range(B)]

            data_t['img_feats'] = data_t['img_feats']
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                        gt_labels_3d[i], gt_bboxes[i],
                                        gt_labels[i], img_metas[i], centers2d[i], depths[i], requires_grad=requires_grad,return_losses=return_losses,**data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value
        return losses


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location


    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi


    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        location = self.prepare_location(img_metas, **data)

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs, _ = self.pts_bbox_head(location, img_metas, None, **data)
            self.train()

        else:
            outs_roi = self.forward_roi_head(location, **data)
            topk_indexes = outs_roi['topk_indexes']
            if self.do_pred:
                outs, pred_feats = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
                outs, pred_loss = self.do_predict(outs, pred_feats, **data)
            else:
                outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)

        if return_losses:
            '''
            ## pred loss 和 其他 loss 一起参与计算
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d) 

            if self.do_pred:
                losses['pred_loss'] = pred_loss if pred_loss is not None else torch.zeros(1)
            return losses
            '''

            if self.do_pred:
                # only 计算 pred loss
                losses = {}
                losses['pred_loss'] = pred_loss if pred_loss is not None else torch.zeros(1)
                return losses
            else:
                loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
                losses = self.pts_bbox_head.loss(*loss_inputs)
                if self.with_img_roi_head:
                    loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
                    losses2d = self.img_roi_head.loss(*loss2d_inputs)
                    losses.update(losses2d) 

                return losses
        else:
            return None


    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            if self.do_pred:
                self.predictor.decoder.do_eval = True
            return self.forward_test(**data)


    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        T = data['img'].size(1)

        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

        if T-self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, T-self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
        else:
            data['img_feats'] = rec_img_feats

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, gt_bboxes,
                        gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)

        return losses
  
  
    def forward_test(self, img_metas, rescale, **data):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            # if key != 'img':
            if key not in self.collect_keys_pred:    # for pred.
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)


    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        topk_indexes = outs_roi['topk_indexes']

        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        if self.do_pred:
            outs, pred_feats = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
            outs, _ = self.do_predict(outs, pred_feats, **data)
        else:
            outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)

        # outs, _ = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results


    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_img_feat(data['img'], 1)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


    def do_predict(self, outs, pred_feats, **data):
        topk_query = pred_feats['topk_query'].transpose(0, 1).contiguous()
        output_reference_point = pred_feats['topk_reference_points']
        output_reference_point_embd = self.pts_bbox_head.query_embedding(pos2posemb3d(output_reference_point)).transpose(0, 1).contiguous()
        memory = pred_feats['memory'].transpose(0, 1).contiguous()
        if self.add_branch:
            query = self.add_branch_mlp(topk_query)
            query = self.add_branch_attention(
                query=query,
                query_pos=output_reference_point_embd,
                key=memory,
                value=memory
            )
            topk_query = query + topk_query

        B = len(data["pred_mapping"])
        batch_pred_outputs, batch_pred_probs = [], []
        batch_loss = torch.zeros(1, device=memory.device)
        for bb in range(B):
            mapping = data["pred_mapping"][bb]
            if not mapping['valid_pred']:
                batch_pred_outputs.append(None)
                batch_pred_probs.append(None)
                continue
            
            future_traj             = data['future_traj'][bb]
            future_traj_is_valid    = data['future_traj_is_valid'][bb]
            past_traj               = data['past_traj'][bb]
            past_traj_is_valid      = data['past_traj_is_valid'][bb]
            labels, labels_is_valid = pred_utils.get_labels_from_reference_points(output_reference_point[bb], future_traj, future_traj_is_valid, past_traj, past_traj_is_valid)

            loss, pred_outputs, _ = self.predictor(agents=topk_query[:, bb].unsqueeze(0),
                                            device=query.device,
                                            labels=[labels],
                                            labels_is_valid=[labels_is_valid],
                                            pred_matrix=[data['pred_matrix'][bb]],
                                            pred_polyline_spans=[data['pred_polyline_spans'][bb]],
                                            pred_mapping=[mapping]
                                            )
            batch_loss += loss
            batch_pred_outputs.append(pred_outputs['pred_outputs'].squeeze())
            batch_pred_probs.append(pred_outputs['pred_probs'].squeeze())
        
        outs.update(
            pred_outputs = batch_pred_outputs,
            pred_probs = batch_pred_probs
        )
        return outs, batch_loss


