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
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from ..predictor import pred_utils
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d


@DETECTORS.register_module()
class Petr3D_Vip3d(MVXTwoStageDetector):
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
                 scores_threshold=0.35,
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
                 add_branch_attention=None,
                 predictor=None,
                 collect_keys_pred=['pred_mapping', 'pred_polyline_spans', 'pred_matrix', 'future_traj', 'future_traj_is_valid', 'past_traj', 'past_traj_is_valid'],
                 ):
        super(Petr3D_Vip3d, self).__init__(pts_voxel_layer, pts_voxel_encoder,
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

        self.scores_threshold =scores_threshold
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
                # self.add_branch_attention = build_from_cfg(pred_utils.get_attention_cfg(), TRANSFORMER_LAYER)
                # self.add_branch_attention = build_from_cfg(pts_bbox_head["transformer"]["decoder"], TRANSFORMER_LAYER_SEQUENCE)
                self.add_branch_attention = build_from_cfg(add_branch_attention, TRANSFORMER_LAYER)


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


    ##########################################################################
    ############################# for test & train ###########################

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2) #1,6,3,320,800
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img) #[6, 3, 320,800]

            img_feats = self.img_backbone(img) #0：[6, 768, 20, 50]    1：[6, 1024, 10, 25]
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)#0：[6, 256, 20, 50]    1：[6, 256, 10, 25]

        BN, C, H, W = img_feats[self.position_level].size()   #6, 256, 20, 50
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN/B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/len_queue), C, H, W)


        return img_feats_reshaped #[1, 1, 6, 256, 20, 50]   B, len_queue, int(BN/B / len_queue), C, H, W


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T, training_mode)
        return img_feats #[1, 1, 6, 256, 20, 50]    B, len_queue, int(BN/B / len_queue), C, H, W


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location #[6, 58, 100, 2]


    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi


    ##########################################################################
    ############################# for train ##################################

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
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads) #[1, 1, 6, 256, 20, 50]

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
        B,T,N,C,H,W= data['img'].shape
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
        location = self.prepare_location(img_metas, **data) #[6, 20, 50, 2]

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
            else:
                outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses, pred_inds = self.pts_bbox_head.loss(*loss_inputs)  # pred_inds：neg的inds是0，pos的inds是对应gt的index+1
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d) 

            if self.do_pred:
                pred_feats.update(pred_inds=pred_inds)
                outs, pred_loss = self.do_predict_train(outs, pred_feats,img_metas, **data)
                losses['pred_loss'] = pred_loss if pred_loss is not None else torch.zeros(1)
            
            return losses
        else:
            return None


    def do_predict_train(self, outs, pred_feats, img_metas, **data):
        B = len(data["img"])
        total_pred_loss = torch.zeros(1, device=data["img_feats"].device, requires_grad=True)
        batch_pred_outputs, batch_pred_probs = [], []
        for b in range(B):
            mapping = data["pred_mapping"][b]
            valid_pred = mapping['valid_pred']
            if not valid_pred:
                batch_pred_outputs.append(None)
                batch_pred_probs.append(None)
                continue
            
            future_traj          = data['future_traj_relative'][b] if self.relative_pred else data['future_traj'][b]
            future_traj_is_valid = data['future_traj_is_valid'][b]
            past_traj            = data['past_traj'][b]
            past_traj_is_valid   = data['past_traj_is_valid'][b]

            pred_query = pred_feats["query"][b][pred_feats["pred_inds"][b]>0].contiguous()
            pred_reference_points = pred_feats['reference_points'][b][pred_feats["pred_inds"][b]>0].contiguous()
            pred_reference_points = pred_utils.reference_points_lidar_to_relative(pred_reference_points, self.pts_bbox_head.pc_range)
            tmp_img_metas = img_metas[b].copy()
            tmp_img_metas['lidar2img'] = data["lidar2img"][b][None,]
            if self.add_branch:
                query = self.add_branch_mlp(pred_query)
                query = self.add_branch_attention(query=query.unsqueeze(1),
                                            reference_points=pred_reference_points.unsqueeze(0),
                                            value=[data["img_feats"][b][None,]],
                                            img_metas=[tmp_img_metas])
                query = query.squeeze(1)
                pred_query = query + pred_query
            if self.agents_layer_0:
                output_embedding = self.agents_layer_mlp_0(pred_query)
            
            pred_inds = pred_feats['pred_inds'][b][torch.nonzero(pred_feats['pred_inds'][b])] - 1  # 减1是因为，本来索引就是+1过的(为了和填充的0区分开)
            labels = np.array([future_traj[i].numpy() for i in pred_inds])
            labels_is_valid = np.array([future_traj_is_valid[i].numpy() for i in pred_inds]).reshape(-1, 12)
        
            kwargs = {'pred_matrix': [data['pred_matrix'][b]],
                      'pred_polyline_spans': [data['pred_polyline_spans'][b]],
                      'pred_mapping': [data['pred_mapping'][b]],}
            
            pred_loss, pred_output, _ = self.predictor(agents=output_embedding.unsqueeze(0), #[1,n,256]
                                                    device=query.device,
                                                    labels=[labels],
                                                    labels_is_valid=[labels_is_valid],
                                                    **kwargs)
            total_pred_loss = total_pred_loss + pred_loss
            batch_pred_outputs.append(pred_output['pred_outputs'].squeeze())
            batch_pred_probs.append(pred_output['pred_probs'].squeeze())


        outs.update(
            pred_outputs = batch_pred_outputs,
            pred_probs = batch_pred_probs
        )
        # print("pred_loss = ", total_pred_loss)
        return outs, total_pred_loss


    ##########################################################################
    ############################# for test ###################################

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


    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_img_feat(data['img'], 1)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


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
            outs.update(pred_feats)
            bbox_list, pred_outs_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
            bbox_list = self.do_predict_test(bbox_list, pred_outs_list, img_metas, **data)
            results = dict(
                        boxes_3d=bbox_list[0][0].to('cpu'),
                        scores_3d=bbox_list[0][1].cpu(),
                        labels_3d=bbox_list[0][2].cpu(),
                        pred_outputs=bbox_list[0][3],
                        pred_probs=bbox_list[0][4])
            return [results]
        else:
            outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)

        # outs, _ = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
        bbox_list,_ = self.pts_bbox_head.get_bboxes(outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results


    def do_predict_test(self, bbox_list, pred_outs_list, img_metas, **data):
        tmp_img_metas = img_metas[0].copy()
        tmp_img_metas['lidar2img'] = data["lidar2img"]

        #process pred_feats
        pred_query = pred_outs_list[0]["pred_query"].contiguous()
        pred_reference_points = pred_outs_list[0]['reference_points'].contiguous()
        pred_reference_points = pred_utils.reference_points_lidar_to_relative(pred_reference_points, self.pts_bbox_head.pc_range)

        if self.add_branch:
            query = self.add_branch_mlp(pred_query)
            query = self.add_branch_attention(query=query.unsqueeze(1),
                                        reference_points=pred_reference_points.unsqueeze(0),
                                        value=[data["img_feats"][0][None,]],
                                        img_metas=[tmp_img_metas])
            query = query.squeeze(1)
            pred_query = query + pred_query
        if self.agents_layer_0:
            output_embedding = self.agents_layer_mlp_0(pred_query)
        
        #全零labels，输出的loss无用
        labels_list = [np.zeros((12, 2))] * len(output_embedding)
        labels_is_valid_list = [np.zeros((12))] * len(output_embedding)

        kwargs = {'pred_matrix': [data['pred_matrix'][0]],
                'pred_polyline_spans': [data['pred_polyline_spans'][0]],
                'pred_mapping': [data['pred_mapping'][0]],}
        
        pred_loss, pred_outputs, _ = self.predictor(agents=output_embedding.unsqueeze(0),   #[1,n,256]
                                                device=query.device,
                                                labels=[np.array(labels_list)],    #list(array(n,12,2))
                                                labels_is_valid=[np.array(labels_is_valid_list)],   #list(array(n,12))
                                                **kwargs)
        
        if self.relative_pred:
            centers_2d = bbox_list[0][0].center[:,:2].cpu().numpy()
            for j in range(len(centers_2d)):
                normalizer = pred_utils.Normalizer(centers_2d[j][0], centers_2d[j][1], 0.0)
                for k in range(len(pred_outputs['pred_outputs'][0][j])):
                    pred_outputs['pred_outputs'][0][j][k] = normalizer(pred_outputs['pred_outputs'][0][j][k], reverse=True)
        
        bbox_list[0].append(pred_outputs['pred_outputs'][0])
        bbox_list[0].append(pred_outputs['pred_probs'][0])
        return bbox_list
    