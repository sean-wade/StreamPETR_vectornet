_base_ = [
    '/workspace/mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '/workspace/mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False) # fix img_norm
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

use_mini = False
do_prediction = True
num_gpus = 1
batch_size = 1
num_workers = 0
num_epochs = 200 if use_mini else 50
base_lr = 2e-4
scores_threshold = 0.35
# num_iters_per_epoch = 323 // (num_gpus * batch_size)
num_iters_per_epoch = 323 // (num_gpus * batch_size) if use_mini else 28130 // (num_gpus * batch_size)
ckpt = 'stream_petr_vov_flash_800_bs2_seq_24e.pth'

# dataset_type = 'CustomNuScenesDataset'
dataset_type = 'StreamPredNuScenesDataset'
data_root = '/mnt/data/dataset/nuScenes/nuscenes_mini/' if use_mini else '/mnt/data/dataset/nuScenes/nuscenes/'

queue_length = 1
num_frame_losses = 1
collect_keys_pred = ['pred_mapping', 'pred_polyline_spans', 'pred_matrix', 'future_traj', 'future_traj_is_valid', 'past_traj', 'past_traj_is_valid', 'future_traj_relative']
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv'] + collect_keys_pred

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

model = dict(
    type='Petr3D_Vip3d',
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    use_grid_mask=True,
    scores_threshold = scores_threshold,

    train_img = False,
    train_roi_head = False,
    train_pts_bbox_head = False,
    train_predictor = True,

    img_backbone=dict(
        type='VoVNetCP', ###use checkpoint to save memory
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    img_roi_head=dict(
        type='FocalHead',
        num_classes=10,
        in_channels=256,
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
        train_cfg=dict(
        assigner2d=dict(
            type='HungarianAssigner2D',
            cls_cost=dict(type='FocalLossCost', weight=2.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0)))
        ),
    pts_bbox_head=dict(
        type='StreamPETRVIP3DHead',
        num_classes=10,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        with_ego_pos=True,
        match_with_velo=False,
        scalar=10, ##noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        do_pred=do_prediction,
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTemporalDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            # type='NMSFreeCoder',
            type='NMSFreeCoderPredict',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=256,    # has to be 256 for now.
            voxel_size=voxel_size,
            num_classes=10,
            score_threshold=0.35
            ), 
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),),

    # prediction from vip3d
    do_pred=do_prediction,
    only_pred_loss=True,
    pred_embed_dims=256,
    relative_pred=True,
    agents_layer_0=True,
    agents_layer_0_num=2,
    add_branch=True,
    add_branch_attention = dict(
        type='DetrTransformerDecoderLayer',
        attn_cfgs=[
            dict(
                type='MultiheadAttention',
                embed_dims=256,
                num_heads=8,
                dropout=0.1),
            dict(
                type='Detr3DCrossAtten',
                pc_range=point_cloud_range,
                num_points=1,
                embed_dims=256,
            )
        ],
        feedforward_channels=512,
        ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                         'ffn', 'norm')),
    predictor=dict(
        hidden_size=128,
        laneGCN=True,
        decoder=dict(
            variety_loss=True,
            variety_loss_prob=True,
            hidden_size=128,
        ),
    ),
    collect_keys_pred=collect_keys_pred,

    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range),)))


file_client_args = dict(backend='disk')


ida_aug_conf = {
        "resize_lim": (0.47, 0.625),
        "final_dim": (320, 800),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='InstanceRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='InstanceNameFilter', classes=class_names),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=True),
    # dict(type='GlobalRotScaleTransImage',
    #         rot_range=[-0.3925, 0.3925],
    #         translation_std=[0, 0, 0],
    #         scale_ratio_range=[0.95, 1.05],
    #         reverse_angle=True,
    #         training=True,
    #         ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
             meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'] + collect_keys,
            meta_keys=('filename', 'ori_shape', 'img_shape','pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=num_workers,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_tracking_streampetr_vip3d_infos_train.pkl',
        num_frame_losses=num_frame_losses,
        seq_split_num=2, # streaming video training
        seq_mode=True, # streaming video training
        mini=use_mini,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        collect_keys_pred = collect_keys_pred,
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root, 
             test_mode=True,
             mini=use_mini, 
             pipeline=test_pipeline, 
             collect_keys=collect_keys + ['img', 'img_metas'], 
             collect_keys_pred = collect_keys_pred,
             queue_length=queue_length, 
             ann_file=data_root + 'nuscenes_tracking_streampetr_vip3d_infos_val.pkl', 
             classes=class_names, 
             modality=input_modality),
    test=dict(type=dataset_type,
              data_root=data_root, 
              test_mode=True,
              mini=use_mini, 
              pipeline=test_pipeline, 
              collect_keys=collect_keys + ['img', 'img_metas'], 
              collect_keys_pred = collect_keys_pred,
              queue_length=queue_length, 
              ann_file=data_root + 'nuscenes_tracking_streampetr_vip3d_infos_val.pkl', 
              classes=class_names, 
              modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )


optimizer = dict(
    type='AdamW', 
    lr=base_lr, # bs 8: 2e-4 || bs 16: 4e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1), # set to 0.1 always better when apply 2D pretrained.
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)

load_from=ckpt
# load_from='work_dirs/mini/E2_stream_petr_vov_flash_800_bs16_wk4_seq_24e/latest.pth'
# load_from=None
resume_from=None
