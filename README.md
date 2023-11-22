<div align="center">
<h1>StreamPETR_vectornet</h1>
<h3>Add trajectory prediction branch for StreamPETR(using vectornet like ViP3D)</h3>
</div>


streampetr 的 CustomNuScenesDataset：
```
    __init__ 中多几个参数

            collect_keys, 
            seq_mode=False, 
            seq_split_num=1, 
            num_frame_losses=1, 
            queue_length=8, 
            random_length=0,

    def _set_sequence_group_flag(self)



    def prepare_train_data(self, index)



    def prepare_test_data(self, index)



    def union2one(self, queue)



    def get_data_info(self, index)



    def __getitem__(self, idx)

```


额外的 helper function：
```
    def invert_matrix_egopose_numpy(egopose)

    def convert_egopose_to_matrix_numpy(rotation, translation)
```


pred_data 多了下面四个 keys:
    ['instance_idx_2_labels', 'pred_matrix', 'polyline_spans', 'mapping']

    instance_idx_2_labels 的 keys:

    dict_keys([794, 796, 797, 800, 801, 802, 803, 804, 805, 806, 807, 809, 810, 813, 814, 815, 816,
                818, 819, 820, 822, 823, 825, 826, 827, 828, 829, 830, 833, 836, 842, 843, 844, 845, 
                846, 847, 848, 850, 851, 853, 854, 855, 858, 859, 860, 861, 864, 865, 866, 868, 869, 870, 871, 873, 874, 875, 876, 821, 824, 863, 834])

        instance_idx_2_labels['794'] 的 keys: 
            ['future_traj',             12x2
            'future_traj_relative',     12x2
            'future_traj_is_valid',     12
            'past_traj',                3x2
            'past_traj_is_valid',       3
            'category',                 3
            'past_boxes'                3x7
            ]


    pred_matrix: 273*128

    polyline_spans: [70 个 slice]

    mapping 的 Keys:
        cur_l2e_r:  1*4
        cur_l2e_t:  1*3
        cur_e2g_r: 1*4
        cur_e2g_t: 1*3
        r_index_2_rotation_and_transform:  {0：{}, 1:{}, 2：{} }
        valid_pred:  True
        instance_inds:  56,
        lanes: L * (X * 2),  L 个 X*2 的 np.array, X不固定
        map_name:  'singapore-hollandvillage'
        timestamp:  1542800853.447313
        timestamp_origin:  1542800853447313
        sample_token:  'f65ffdc408fb4a0c8ef0d1614b47dce8'
        scene_id:  'scene-1094'
        same_scene:  True
        index:  252



pts_bbox_head.code_weights
pts_bbox_head.match_costs
pts_bbox_head.pc_range
pts_bbox_head.position_range
pts_bbox_head.coords_d
pts_bbox_head.cls_branches.0.0.weight
pts_bbox_head.cls_branches.0.0.bias
pts_bbox_head.cls_branches.0.1.weight
pts_bbox_head.cls_branches.0.1.bias
pts_bbox_head.cls_branches.0.3.weight
pts_bbox_head.cls_branches.0.3.bias
pts_bbox_head.cls_branches.0.4.weight
pts_bbox_head.cls_branches.0.4.bias
pts_bbox_head.cls_branches.0.6.weight
pts_bbox_head.cls_branches.0.6.bias
pts_bbox_head.reg_branches.0.0.weight
pts_bbox_head.reg_branches.0.0.bias
pts_bbox_head.reg_branches.0.2.weight
pts_bbox_head.reg_branches.0.2.bias
pts_bbox_head.reg_branches.0.4.weight
pts_bbox_head.reg_branches.0.4.bias
pts_bbox_head.position_encoder.0.weight
pts_bbox_head.position_encoder.0.bias
pts_bbox_head.position_encoder.2.weight
pts_bbox_head.position_encoder.2.bias
pts_bbox_head.memory_embed.0.weight
pts_bbox_head.memory_embed.0.bias
pts_bbox_head.memory_embed.2.weight
pts_bbox_head.memory_embed.2.bias
pts_bbox_head.featurized_pe.conv_reduce.weight
pts_bbox_head.featurized_pe.conv_reduce.bias
pts_bbox_head.featurized_pe.conv_expand.weight
pts_bbox_head.featurized_pe.conv_expand.bias
pts_bbox_head.reference_points.weight
pts_bbox_head.pseudo_reference_points.weight
pts_bbox_head.query_embedding.0.weight
pts_bbox_head.query_embedding.0.bias
pts_bbox_head.query_embedding.2.weight
pts_bbox_head.query_embedding.2.bias
pts_bbox_head.spatial_alignment.reduce.0.weight
pts_bbox_head.spatial_alignment.reduce.0.bias
pts_bbox_head.spatial_alignment.gamma.weight
pts_bbox_head.spatial_alignment.gamma.bias
pts_bbox_head.spatial_alignment.beta.weight
pts_bbox_head.spatial_alignment.beta.bias
pts_bbox_head.time_embedding.0.weight
pts_bbox_head.time_embedding.0.bias
pts_bbox_head.time_embedding.1.weight
pts_bbox_head.time_embedding.1.bias
pts_bbox_head.ego_pose_pe.reduce.0.weight
pts_bbox_head.ego_pose_pe.reduce.0.bias
pts_bbox_head.ego_pose_pe.gamma.weight
pts_bbox_head.ego_pose_pe.gamma.bias
pts_bbox_head.ego_pose_pe.beta.weight
pts_bbox_head.ego_pose_pe.beta.bias
pts_bbox_head.ego_pose_memory.reduce.0.weight
pts_bbox_head.ego_pose_memory.reduce.0.bias
pts_bbox_head.ego_pose_memory.gamma.weight
pts_bbox_head.ego_pose_memory.gamma.bias
pts_bbox_head.ego_pose_memory.beta.weight
pts_bbox_head.ego_pose_memory.beta.bias
pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.0.attentions.1.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.0.attentions.1.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.0.attentions.1.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.0.attentions.1.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.0.0.weight
pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.0.0.bias
pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.1.weight
pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.1.bias
pts_bbox_head.transformer.decoder.layers.0.norms.0.weight
pts_bbox_head.transformer.decoder.layers.0.norms.0.bias
pts_bbox_head.transformer.decoder.layers.0.norms.1.weight
pts_bbox_head.transformer.decoder.layers.0.norms.1.bias
pts_bbox_head.transformer.decoder.layers.0.norms.2.weight
pts_bbox_head.transformer.decoder.layers.0.norms.2.bias
pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.1.attentions.1.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.1.attentions.1.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.1.attentions.1.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.1.attentions.1.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.0.0.weight
pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.0.0.bias
pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.1.weight
pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.1.bias
pts_bbox_head.transformer.decoder.layers.1.norms.0.weight
pts_bbox_head.transformer.decoder.layers.1.norms.0.bias
pts_bbox_head.transformer.decoder.layers.1.norms.1.weight
pts_bbox_head.transformer.decoder.layers.1.norms.1.bias
pts_bbox_head.transformer.decoder.layers.1.norms.2.weight
pts_bbox_head.transformer.decoder.layers.1.norms.2.bias
pts_bbox_head.transformer.decoder.layers.2.attentions.0.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.2.attentions.0.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.2.attentions.0.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.2.attentions.0.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.2.attentions.1.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.2.attentions.1.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.2.attentions.1.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.2.attentions.1.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.2.ffns.0.layers.0.0.weight
pts_bbox_head.transformer.decoder.layers.2.ffns.0.layers.0.0.bias
pts_bbox_head.transformer.decoder.layers.2.ffns.0.layers.1.weight
pts_bbox_head.transformer.decoder.layers.2.ffns.0.layers.1.bias
pts_bbox_head.transformer.decoder.layers.2.norms.0.weight
pts_bbox_head.transformer.decoder.layers.2.norms.0.bias
pts_bbox_head.transformer.decoder.layers.2.norms.1.weight
pts_bbox_head.transformer.decoder.layers.2.norms.1.bias
pts_bbox_head.transformer.decoder.layers.2.norms.2.weight
pts_bbox_head.transformer.decoder.layers.2.norms.2.bias
pts_bbox_head.transformer.decoder.layers.3.attentions.0.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.3.attentions.0.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.3.attentions.0.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.3.attentions.0.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.3.attentions.1.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.3.attentions.1.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.3.attentions.1.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.3.attentions.1.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.3.ffns.0.layers.0.0.weight
pts_bbox_head.transformer.decoder.layers.3.ffns.0.layers.0.0.bias
pts_bbox_head.transformer.decoder.layers.3.ffns.0.layers.1.weight
pts_bbox_head.transformer.decoder.layers.3.ffns.0.layers.1.bias
pts_bbox_head.transformer.decoder.layers.3.norms.0.weight
pts_bbox_head.transformer.decoder.layers.3.norms.0.bias
pts_bbox_head.transformer.decoder.layers.3.norms.1.weight
pts_bbox_head.transformer.decoder.layers.3.norms.1.bias
pts_bbox_head.transformer.decoder.layers.3.norms.2.weight
pts_bbox_head.transformer.decoder.layers.3.norms.2.bias
pts_bbox_head.transformer.decoder.layers.4.attentions.0.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.4.attentions.0.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.4.attentions.0.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.4.attentions.0.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.4.attentions.1.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.4.attentions.1.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.4.attentions.1.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.4.attentions.1.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.4.ffns.0.layers.0.0.weight
pts_bbox_head.transformer.decoder.layers.4.ffns.0.layers.0.0.bias
pts_bbox_head.transformer.decoder.layers.4.ffns.0.layers.1.weight
pts_bbox_head.transformer.decoder.layers.4.ffns.0.layers.1.bias
pts_bbox_head.transformer.decoder.layers.4.norms.0.weight
pts_bbox_head.transformer.decoder.layers.4.norms.0.bias
pts_bbox_head.transformer.decoder.layers.4.norms.1.weight
pts_bbox_head.transformer.decoder.layers.4.norms.1.bias
pts_bbox_head.transformer.decoder.layers.4.norms.2.weight
pts_bbox_head.transformer.decoder.layers.4.norms.2.bias
pts_bbox_head.transformer.decoder.layers.5.attentions.0.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.5.attentions.0.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.5.attentions.0.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.5.attentions.0.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.5.attentions.1.attn.in_proj_weight
pts_bbox_head.transformer.decoder.layers.5.attentions.1.attn.in_proj_bias
pts_bbox_head.transformer.decoder.layers.5.attentions.1.attn.out_proj.weight
pts_bbox_head.transformer.decoder.layers.5.attentions.1.attn.out_proj.bias
pts_bbox_head.transformer.decoder.layers.5.ffns.0.layers.0.0.weight
pts_bbox_head.transformer.decoder.layers.5.ffns.0.layers.0.0.bias
pts_bbox_head.transformer.decoder.layers.5.ffns.0.layers.1.weight
pts_bbox_head.transformer.decoder.layers.5.ffns.0.layers.1.bias
pts_bbox_head.transformer.decoder.layers.5.norms.0.weight
pts_bbox_head.transformer.decoder.layers.5.norms.0.bias
pts_bbox_head.transformer.decoder.layers.5.norms.1.weight
pts_bbox_head.transformer.decoder.layers.5.norms.1.bias
pts_bbox_head.transformer.decoder.layers.5.norms.2.weight
pts_bbox_head.transformer.decoder.layers.5.norms.2.bias
pts_bbox_head.transformer.decoder.post_norm.weight
pts_bbox_head.transformer.decoder.post_norm.bias
img_backbone.stem.stem_1/conv.weight
img_backbone.stem.stem_1/norm.weight
img_backbone.stem.stem_1/norm.bias
img_backbone.stem.stem_2/conv.weight
img_backbone.stem.stem_2/norm.weight
img_backbone.stem.stem_2/norm.bias
img_backbone.stem.stem_3/conv.weight
img_backbone.stem.stem_3/norm.weight
img_backbone.stem.stem_3/norm.bias
img_backbone.stage2.OSA2_1.layers.0.OSA2_1_0/conv.weight
img_backbone.stage2.OSA2_1.layers.0.OSA2_1_0/norm.weight
img_backbone.stage2.OSA2_1.layers.0.OSA2_1_0/norm.bias
img_backbone.stage2.OSA2_1.layers.1.OSA2_1_1/conv.weight
img_backbone.stage2.OSA2_1.layers.1.OSA2_1_1/norm.weight
img_backbone.stage2.OSA2_1.layers.1.OSA2_1_1/norm.bias
img_backbone.stage2.OSA2_1.layers.2.OSA2_1_2/conv.weight
img_backbone.stage2.OSA2_1.layers.2.OSA2_1_2/norm.weight
img_backbone.stage2.OSA2_1.layers.2.OSA2_1_2/norm.bias
img_backbone.stage2.OSA2_1.layers.3.OSA2_1_3/conv.weight
img_backbone.stage2.OSA2_1.layers.3.OSA2_1_3/norm.weight
img_backbone.stage2.OSA2_1.layers.3.OSA2_1_3/norm.bias
img_backbone.stage2.OSA2_1.layers.4.OSA2_1_4/conv.weight
img_backbone.stage2.OSA2_1.layers.4.OSA2_1_4/norm.weight
img_backbone.stage2.OSA2_1.layers.4.OSA2_1_4/norm.bias
img_backbone.stage2.OSA2_1.concat.OSA2_1_concat/conv.weight
img_backbone.stage2.OSA2_1.concat.OSA2_1_concat/norm.weight
img_backbone.stage2.OSA2_1.concat.OSA2_1_concat/norm.bias
img_backbone.stage2.OSA2_1.ese.fc.weight
img_backbone.stage2.OSA2_1.ese.fc.bias
img_backbone.stage3.OSA3_1.layers.0.OSA3_1_0/conv.weight
img_backbone.stage3.OSA3_1.layers.0.OSA3_1_0/norm.weight
img_backbone.stage3.OSA3_1.layers.0.OSA3_1_0/norm.bias
img_backbone.stage3.OSA3_1.layers.1.OSA3_1_1/conv.weight
img_backbone.stage3.OSA3_1.layers.1.OSA3_1_1/norm.weight
img_backbone.stage3.OSA3_1.layers.1.OSA3_1_1/norm.bias
img_backbone.stage3.OSA3_1.layers.2.OSA3_1_2/conv.weight
img_backbone.stage3.OSA3_1.layers.2.OSA3_1_2/norm.weight
img_backbone.stage3.OSA3_1.layers.2.OSA3_1_2/norm.bias
img_backbone.stage3.OSA3_1.layers.3.OSA3_1_3/conv.weight
img_backbone.stage3.OSA3_1.layers.3.OSA3_1_3/norm.weight
img_backbone.stage3.OSA3_1.layers.3.OSA3_1_3/norm.bias
img_backbone.stage3.OSA3_1.layers.4.OSA3_1_4/conv.weight
img_backbone.stage3.OSA3_1.layers.4.OSA3_1_4/norm.weight
img_backbone.stage3.OSA3_1.layers.4.OSA3_1_4/norm.bias
img_backbone.stage3.OSA3_1.concat.OSA3_1_concat/conv.weight
img_backbone.stage3.OSA3_1.concat.OSA3_1_concat/norm.weight
img_backbone.stage3.OSA3_1.concat.OSA3_1_concat/norm.bias
img_backbone.stage3.OSA3_1.ese.fc.weight
img_backbone.stage3.OSA3_1.ese.fc.bias
img_backbone.stage3.OSA3_2.layers.0.OSA3_2_0/conv.weight
img_backbone.stage3.OSA3_2.layers.0.OSA3_2_0/norm.weight
img_backbone.stage3.OSA3_2.layers.0.OSA3_2_0/norm.bias
img_backbone.stage3.OSA3_2.layers.1.OSA3_2_1/conv.weight
img_backbone.stage3.OSA3_2.layers.1.OSA3_2_1/norm.weight
img_backbone.stage3.OSA3_2.layers.1.OSA3_2_1/norm.bias
img_backbone.stage3.OSA3_2.layers.2.OSA3_2_2/conv.weight
img_backbone.stage3.OSA3_2.layers.2.OSA3_2_2/norm.weight
img_backbone.stage3.OSA3_2.layers.2.OSA3_2_2/norm.bias
img_backbone.stage3.OSA3_2.layers.3.OSA3_2_3/conv.weight
img_backbone.stage3.OSA3_2.layers.3.OSA3_2_3/norm.weight
img_backbone.stage3.OSA3_2.layers.3.OSA3_2_3/norm.bias
img_backbone.stage3.OSA3_2.layers.4.OSA3_2_4/conv.weight
img_backbone.stage3.OSA3_2.layers.4.OSA3_2_4/norm.weight
img_backbone.stage3.OSA3_2.layers.4.OSA3_2_4/norm.bias
img_backbone.stage3.OSA3_2.concat.OSA3_2_concat/conv.weight
img_backbone.stage3.OSA3_2.concat.OSA3_2_concat/norm.weight
img_backbone.stage3.OSA3_2.concat.OSA3_2_concat/norm.bias
img_backbone.stage3.OSA3_2.ese.fc.weight
img_backbone.stage3.OSA3_2.ese.fc.bias
img_backbone.stage3.OSA3_3.layers.0.OSA3_3_0/conv.weight
img_backbone.stage3.OSA3_3.layers.0.OSA3_3_0/norm.weight
img_backbone.stage3.OSA3_3.layers.0.OSA3_3_0/norm.bias
img_backbone.stage3.OSA3_3.layers.1.OSA3_3_1/conv.weight
img_backbone.stage3.OSA3_3.layers.1.OSA3_3_1/norm.weight
img_backbone.stage3.OSA3_3.layers.1.OSA3_3_1/norm.bias
img_backbone.stage3.OSA3_3.layers.2.OSA3_3_2/conv.weight
img_backbone.stage3.OSA3_3.layers.2.OSA3_3_2/norm.weight
img_backbone.stage3.OSA3_3.layers.2.OSA3_3_2/norm.bias
img_backbone.stage3.OSA3_3.layers.3.OSA3_3_3/conv.weight
img_backbone.stage3.OSA3_3.layers.3.OSA3_3_3/norm.weight
img_backbone.stage3.OSA3_3.layers.3.OSA3_3_3/norm.bias
img_backbone.stage3.OSA3_3.layers.4.OSA3_3_4/conv.weight
img_backbone.stage3.OSA3_3.layers.4.OSA3_3_4/norm.weight
img_backbone.stage3.OSA3_3.layers.4.OSA3_3_4/norm.bias
img_backbone.stage3.OSA3_3.concat.OSA3_3_concat/conv.weight
img_backbone.stage3.OSA3_3.concat.OSA3_3_concat/norm.weight
img_backbone.stage3.OSA3_3.concat.OSA3_3_concat/norm.bias
img_backbone.stage3.OSA3_3.ese.fc.weight
img_backbone.stage3.OSA3_3.ese.fc.bias
img_backbone.stage4.OSA4_1.layers.0.OSA4_1_0/conv.weight
img_backbone.stage4.OSA4_1.layers.0.OSA4_1_0/norm.weight
img_backbone.stage4.OSA4_1.layers.0.OSA4_1_0/norm.bias
img_backbone.stage4.OSA4_1.layers.1.OSA4_1_1/conv.weight
img_backbone.stage4.OSA4_1.layers.1.OSA4_1_1/norm.weight
img_backbone.stage4.OSA4_1.layers.1.OSA4_1_1/norm.bias
img_backbone.stage4.OSA4_1.layers.2.OSA4_1_2/conv.weight
img_backbone.stage4.OSA4_1.layers.2.OSA4_1_2/norm.weight
img_backbone.stage4.OSA4_1.layers.2.OSA4_1_2/norm.bias
img_backbone.stage4.OSA4_1.layers.3.OSA4_1_3/conv.weight
img_backbone.stage4.OSA4_1.layers.3.OSA4_1_3/norm.weight
img_backbone.stage4.OSA4_1.layers.3.OSA4_1_3/norm.bias
img_backbone.stage4.OSA4_1.layers.4.OSA4_1_4/conv.weight
img_backbone.stage4.OSA4_1.layers.4.OSA4_1_4/norm.weight
img_backbone.stage4.OSA4_1.layers.4.OSA4_1_4/norm.bias
img_backbone.stage4.OSA4_1.concat.OSA4_1_concat/conv.weight
img_backbone.stage4.OSA4_1.concat.OSA4_1_concat/norm.weight
img_backbone.stage4.OSA4_1.concat.OSA4_1_concat/norm.bias
img_backbone.stage4.OSA4_1.ese.fc.weight
img_backbone.stage4.OSA4_1.ese.fc.bias
img_backbone.stage4.OSA4_2.layers.0.OSA4_2_0/conv.weight
img_backbone.stage4.OSA4_2.layers.0.OSA4_2_0/norm.weight
img_backbone.stage4.OSA4_2.layers.0.OSA4_2_0/norm.bias
img_backbone.stage4.OSA4_2.layers.1.OSA4_2_1/conv.weight
img_backbone.stage4.OSA4_2.layers.1.OSA4_2_1/norm.weight
img_backbone.stage4.OSA4_2.layers.1.OSA4_2_1/norm.bias
img_backbone.stage4.OSA4_2.layers.2.OSA4_2_2/conv.weight
img_backbone.stage4.OSA4_2.layers.2.OSA4_2_2/norm.weight
img_backbone.stage4.OSA4_2.layers.2.OSA4_2_2/norm.bias
img_backbone.stage4.OSA4_2.layers.3.OSA4_2_3/conv.weight
img_backbone.stage4.OSA4_2.layers.3.OSA4_2_3/norm.weight
img_backbone.stage4.OSA4_2.layers.3.OSA4_2_3/norm.bias
img_backbone.stage4.OSA4_2.layers.4.OSA4_2_4/conv.weight
img_backbone.stage4.OSA4_2.layers.4.OSA4_2_4/norm.weight
img_backbone.stage4.OSA4_2.layers.4.OSA4_2_4/norm.bias
img_backbone.stage4.OSA4_2.concat.OSA4_2_concat/conv.weight
img_backbone.stage4.OSA4_2.concat.OSA4_2_concat/norm.weight
img_backbone.stage4.OSA4_2.concat.OSA4_2_concat/norm.bias
img_backbone.stage4.OSA4_2.ese.fc.weight
img_backbone.stage4.OSA4_2.ese.fc.bias
img_backbone.stage4.OSA4_3.layers.0.OSA4_3_0/conv.weight
img_backbone.stage4.OSA4_3.layers.0.OSA4_3_0/norm.weight
img_backbone.stage4.OSA4_3.layers.0.OSA4_3_0/norm.bias
img_backbone.stage4.OSA4_3.layers.1.OSA4_3_1/conv.weight
img_backbone.stage4.OSA4_3.layers.1.OSA4_3_1/norm.weight
img_backbone.stage4.OSA4_3.layers.1.OSA4_3_1/norm.bias
img_backbone.stage4.OSA4_3.layers.2.OSA4_3_2/conv.weight
img_backbone.stage4.OSA4_3.layers.2.OSA4_3_2/norm.weight
img_backbone.stage4.OSA4_3.layers.2.OSA4_3_2/norm.bias
img_backbone.stage4.OSA4_3.layers.3.OSA4_3_3/conv.weight
img_backbone.stage4.OSA4_3.layers.3.OSA4_3_3/norm.weight
img_backbone.stage4.OSA4_3.layers.3.OSA4_3_3/norm.bias
img_backbone.stage4.OSA4_3.layers.4.OSA4_3_4/conv.weight
img_backbone.stage4.OSA4_3.layers.4.OSA4_3_4/norm.weight
img_backbone.stage4.OSA4_3.layers.4.OSA4_3_4/norm.bias
img_backbone.stage4.OSA4_3.concat.OSA4_3_concat/conv.weight
img_backbone.stage4.OSA4_3.concat.OSA4_3_concat/norm.weight
img_backbone.stage4.OSA4_3.concat.OSA4_3_concat/norm.bias
img_backbone.stage4.OSA4_3.ese.fc.weight
img_backbone.stage4.OSA4_3.ese.fc.bias
img_backbone.stage4.OSA4_4.layers.0.OSA4_4_0/conv.weight
img_backbone.stage4.OSA4_4.layers.0.OSA4_4_0/norm.weight
img_backbone.stage4.OSA4_4.layers.0.OSA4_4_0/norm.bias
img_backbone.stage4.OSA4_4.layers.1.OSA4_4_1/conv.weight
img_backbone.stage4.OSA4_4.layers.1.OSA4_4_1/norm.weight
img_backbone.stage4.OSA4_4.layers.1.OSA4_4_1/norm.bias
img_backbone.stage4.OSA4_4.layers.2.OSA4_4_2/conv.weight
img_backbone.stage4.OSA4_4.layers.2.OSA4_4_2/norm.weight
img_backbone.stage4.OSA4_4.layers.2.OSA4_4_2/norm.bias
img_backbone.stage4.OSA4_4.layers.3.OSA4_4_3/conv.weight
img_backbone.stage4.OSA4_4.layers.3.OSA4_4_3/norm.weight
img_backbone.stage4.OSA4_4.layers.3.OSA4_4_3/norm.bias
img_backbone.stage4.OSA4_4.layers.4.OSA4_4_4/conv.weight
img_backbone.stage4.OSA4_4.layers.4.OSA4_4_4/norm.weight
img_backbone.stage4.OSA4_4.layers.4.OSA4_4_4/norm.bias
img_backbone.stage4.OSA4_4.concat.OSA4_4_concat/conv.weight
img_backbone.stage4.OSA4_4.concat.OSA4_4_concat/norm.weight
img_backbone.stage4.OSA4_4.concat.OSA4_4_concat/norm.bias
img_backbone.stage4.OSA4_4.ese.fc.weight
img_backbone.stage4.OSA4_4.ese.fc.bias
img_backbone.stage4.OSA4_5.layers.0.OSA4_5_0/conv.weight
img_backbone.stage4.OSA4_5.layers.0.OSA4_5_0/norm.weight
img_backbone.stage4.OSA4_5.layers.0.OSA4_5_0/norm.bias
img_backbone.stage4.OSA4_5.layers.1.OSA4_5_1/conv.weight
img_backbone.stage4.OSA4_5.layers.1.OSA4_5_1/norm.weight
img_backbone.stage4.OSA4_5.layers.1.OSA4_5_1/norm.bias
img_backbone.stage4.OSA4_5.layers.2.OSA4_5_2/conv.weight
img_backbone.stage4.OSA4_5.layers.2.OSA4_5_2/norm.weight
img_backbone.stage4.OSA4_5.layers.2.OSA4_5_2/norm.bias
img_backbone.stage4.OSA4_5.layers.3.OSA4_5_3/conv.weight
img_backbone.stage4.OSA4_5.layers.3.OSA4_5_3/norm.weight
img_backbone.stage4.OSA4_5.layers.3.OSA4_5_3/norm.bias
img_backbone.stage4.OSA4_5.layers.4.OSA4_5_4/conv.weight
img_backbone.stage4.OSA4_5.layers.4.OSA4_5_4/norm.weight
img_backbone.stage4.OSA4_5.layers.4.OSA4_5_4/norm.bias
img_backbone.stage4.OSA4_5.concat.OSA4_5_concat/conv.weight
img_backbone.stage4.OSA4_5.concat.OSA4_5_concat/norm.weight
img_backbone.stage4.OSA4_5.concat.OSA4_5_concat/norm.bias
img_backbone.stage4.OSA4_5.ese.fc.weight
img_backbone.stage4.OSA4_5.ese.fc.bias
img_backbone.stage4.OSA4_6.layers.0.OSA4_6_0/conv.weight
img_backbone.stage4.OSA4_6.layers.0.OSA4_6_0/norm.weight
img_backbone.stage4.OSA4_6.layers.0.OSA4_6_0/norm.bias
img_backbone.stage4.OSA4_6.layers.1.OSA4_6_1/conv.weight
img_backbone.stage4.OSA4_6.layers.1.OSA4_6_1/norm.weight
img_backbone.stage4.OSA4_6.layers.1.OSA4_6_1/norm.bias
img_backbone.stage4.OSA4_6.layers.2.OSA4_6_2/conv.weight
img_backbone.stage4.OSA4_6.layers.2.OSA4_6_2/norm.weight
img_backbone.stage4.OSA4_6.layers.2.OSA4_6_2/norm.bias
img_backbone.stage4.OSA4_6.layers.3.OSA4_6_3/conv.weight
img_backbone.stage4.OSA4_6.layers.3.OSA4_6_3/norm.weight
img_backbone.stage4.OSA4_6.layers.3.OSA4_6_3/norm.bias
img_backbone.stage4.OSA4_6.layers.4.OSA4_6_4/conv.weight
img_backbone.stage4.OSA4_6.layers.4.OSA4_6_4/norm.weight
img_backbone.stage4.OSA4_6.layers.4.OSA4_6_4/norm.bias
img_backbone.stage4.OSA4_6.concat.OSA4_6_concat/conv.weight
img_backbone.stage4.OSA4_6.concat.OSA4_6_concat/norm.weight
img_backbone.stage4.OSA4_6.concat.OSA4_6_concat/norm.bias
img_backbone.stage4.OSA4_6.ese.fc.weight
img_backbone.stage4.OSA4_6.ese.fc.bias
img_backbone.stage4.OSA4_7.layers.0.OSA4_7_0/conv.weight
img_backbone.stage4.OSA4_7.layers.0.OSA4_7_0/norm.weight
img_backbone.stage4.OSA4_7.layers.0.OSA4_7_0/norm.bias
img_backbone.stage4.OSA4_7.layers.1.OSA4_7_1/conv.weight
img_backbone.stage4.OSA4_7.layers.1.OSA4_7_1/norm.weight
img_backbone.stage4.OSA4_7.layers.1.OSA4_7_1/norm.bias
img_backbone.stage4.OSA4_7.layers.2.OSA4_7_2/conv.weight
img_backbone.stage4.OSA4_7.layers.2.OSA4_7_2/norm.weight
img_backbone.stage4.OSA4_7.layers.2.OSA4_7_2/norm.bias
img_backbone.stage4.OSA4_7.layers.3.OSA4_7_3/conv.weight
img_backbone.stage4.OSA4_7.layers.3.OSA4_7_3/norm.weight
img_backbone.stage4.OSA4_7.layers.3.OSA4_7_3/norm.bias
img_backbone.stage4.OSA4_7.layers.4.OSA4_7_4/conv.weight
img_backbone.stage4.OSA4_7.layers.4.OSA4_7_4/norm.weight
img_backbone.stage4.OSA4_7.layers.4.OSA4_7_4/norm.bias
img_backbone.stage4.OSA4_7.concat.OSA4_7_concat/conv.weight
img_backbone.stage4.OSA4_7.concat.OSA4_7_concat/norm.weight
img_backbone.stage4.OSA4_7.concat.OSA4_7_concat/norm.bias
img_backbone.stage4.OSA4_7.ese.fc.weight
img_backbone.stage4.OSA4_7.ese.fc.bias
img_backbone.stage4.OSA4_8.layers.0.OSA4_8_0/conv.weight
img_backbone.stage4.OSA4_8.layers.0.OSA4_8_0/norm.weight
img_backbone.stage4.OSA4_8.layers.0.OSA4_8_0/norm.bias
img_backbone.stage4.OSA4_8.layers.1.OSA4_8_1/conv.weight
img_backbone.stage4.OSA4_8.layers.1.OSA4_8_1/norm.weight
img_backbone.stage4.OSA4_8.layers.1.OSA4_8_1/norm.bias
img_backbone.stage4.OSA4_8.layers.2.OSA4_8_2/conv.weight
img_backbone.stage4.OSA4_8.layers.2.OSA4_8_2/norm.weight
img_backbone.stage4.OSA4_8.layers.2.OSA4_8_2/norm.bias
img_backbone.stage4.OSA4_8.layers.3.OSA4_8_3/conv.weight
img_backbone.stage4.OSA4_8.layers.3.OSA4_8_3/norm.weight
img_backbone.stage4.OSA4_8.layers.3.OSA4_8_3/norm.bias
img_backbone.stage4.OSA4_8.layers.4.OSA4_8_4/conv.weight
img_backbone.stage4.OSA4_8.layers.4.OSA4_8_4/norm.weight
img_backbone.stage4.OSA4_8.layers.4.OSA4_8_4/norm.bias
img_backbone.stage4.OSA4_8.concat.OSA4_8_concat/conv.weight
img_backbone.stage4.OSA4_8.concat.OSA4_8_concat/norm.weight
img_backbone.stage4.OSA4_8.concat.OSA4_8_concat/norm.bias
img_backbone.stage4.OSA4_8.ese.fc.weight
img_backbone.stage4.OSA4_8.ese.fc.bias
img_backbone.stage4.OSA4_9.layers.0.OSA4_9_0/conv.weight
img_backbone.stage4.OSA4_9.layers.0.OSA4_9_0/norm.weight
img_backbone.stage4.OSA4_9.layers.0.OSA4_9_0/norm.bias
img_backbone.stage4.OSA4_9.layers.1.OSA4_9_1/conv.weight
img_backbone.stage4.OSA4_9.layers.1.OSA4_9_1/norm.weight
img_backbone.stage4.OSA4_9.layers.1.OSA4_9_1/norm.bias
img_backbone.stage4.OSA4_9.layers.2.OSA4_9_2/conv.weight
img_backbone.stage4.OSA4_9.layers.2.OSA4_9_2/norm.weight
img_backbone.stage4.OSA4_9.layers.2.OSA4_9_2/norm.bias
img_backbone.stage4.OSA4_9.layers.3.OSA4_9_3/conv.weight
img_backbone.stage4.OSA4_9.layers.3.OSA4_9_3/norm.weight
img_backbone.stage4.OSA4_9.layers.3.OSA4_9_3/norm.bias
img_backbone.stage4.OSA4_9.layers.4.OSA4_9_4/conv.weight
img_backbone.stage4.OSA4_9.layers.4.OSA4_9_4/norm.weight
img_backbone.stage4.OSA4_9.layers.4.OSA4_9_4/norm.bias
img_backbone.stage4.OSA4_9.concat.OSA4_9_concat/conv.weight
img_backbone.stage4.OSA4_9.concat.OSA4_9_concat/norm.weight
img_backbone.stage4.OSA4_9.concat.OSA4_9_concat/norm.bias
img_backbone.stage4.OSA4_9.ese.fc.weight
img_backbone.stage4.OSA4_9.ese.fc.bias
img_backbone.stage5.OSA5_1.layers.0.OSA5_1_0/conv.weight
img_backbone.stage5.OSA5_1.layers.0.OSA5_1_0/norm.weight
img_backbone.stage5.OSA5_1.layers.0.OSA5_1_0/norm.bias
img_backbone.stage5.OSA5_1.layers.1.OSA5_1_1/conv.weight
img_backbone.stage5.OSA5_1.layers.1.OSA5_1_1/norm.weight
img_backbone.stage5.OSA5_1.layers.1.OSA5_1_1/norm.bias
img_backbone.stage5.OSA5_1.layers.2.OSA5_1_2/conv.weight
img_backbone.stage5.OSA5_1.layers.2.OSA5_1_2/norm.weight
img_backbone.stage5.OSA5_1.layers.2.OSA5_1_2/norm.bias
img_backbone.stage5.OSA5_1.layers.3.OSA5_1_3/conv.weight
img_backbone.stage5.OSA5_1.layers.3.OSA5_1_3/norm.weight
img_backbone.stage5.OSA5_1.layers.3.OSA5_1_3/norm.bias
img_backbone.stage5.OSA5_1.layers.4.OSA5_1_4/conv.weight
img_backbone.stage5.OSA5_1.layers.4.OSA5_1_4/norm.weight
img_backbone.stage5.OSA5_1.layers.4.OSA5_1_4/norm.bias
img_backbone.stage5.OSA5_1.concat.OSA5_1_concat/conv.weight
img_backbone.stage5.OSA5_1.concat.OSA5_1_concat/norm.weight
img_backbone.stage5.OSA5_1.concat.OSA5_1_concat/norm.bias
img_backbone.stage5.OSA5_1.ese.fc.weight
img_backbone.stage5.OSA5_1.ese.fc.bias
img_backbone.stage5.OSA5_2.layers.0.OSA5_2_0/conv.weight
img_backbone.stage5.OSA5_2.layers.0.OSA5_2_0/norm.weight
img_backbone.stage5.OSA5_2.layers.0.OSA5_2_0/norm.bias
img_backbone.stage5.OSA5_2.layers.1.OSA5_2_1/conv.weight
img_backbone.stage5.OSA5_2.layers.1.OSA5_2_1/norm.weight
img_backbone.stage5.OSA5_2.layers.1.OSA5_2_1/norm.bias
img_backbone.stage5.OSA5_2.layers.2.OSA5_2_2/conv.weight
img_backbone.stage5.OSA5_2.layers.2.OSA5_2_2/norm.weight
img_backbone.stage5.OSA5_2.layers.2.OSA5_2_2/norm.bias
img_backbone.stage5.OSA5_2.layers.3.OSA5_2_3/conv.weight
img_backbone.stage5.OSA5_2.layers.3.OSA5_2_3/norm.weight
img_backbone.stage5.OSA5_2.layers.3.OSA5_2_3/norm.bias
img_backbone.stage5.OSA5_2.layers.4.OSA5_2_4/conv.weight
img_backbone.stage5.OSA5_2.layers.4.OSA5_2_4/norm.weight
img_backbone.stage5.OSA5_2.layers.4.OSA5_2_4/norm.bias
img_backbone.stage5.OSA5_2.concat.OSA5_2_concat/conv.weight
img_backbone.stage5.OSA5_2.concat.OSA5_2_concat/norm.weight
img_backbone.stage5.OSA5_2.concat.OSA5_2_concat/norm.bias
img_backbone.stage5.OSA5_2.ese.fc.weight
img_backbone.stage5.OSA5_2.ese.fc.bias
img_backbone.stage5.OSA5_3.layers.0.OSA5_3_0/conv.weight
img_backbone.stage5.OSA5_3.layers.0.OSA5_3_0/norm.weight
img_backbone.stage5.OSA5_3.layers.0.OSA5_3_0/norm.bias
img_backbone.stage5.OSA5_3.layers.1.OSA5_3_1/conv.weight
img_backbone.stage5.OSA5_3.layers.1.OSA5_3_1/norm.weight
img_backbone.stage5.OSA5_3.layers.1.OSA5_3_1/norm.bias
img_backbone.stage5.OSA5_3.layers.2.OSA5_3_2/conv.weight
img_backbone.stage5.OSA5_3.layers.2.OSA5_3_2/norm.weight
img_backbone.stage5.OSA5_3.layers.2.OSA5_3_2/norm.bias
img_backbone.stage5.OSA5_3.layers.3.OSA5_3_3/conv.weight
img_backbone.stage5.OSA5_3.layers.3.OSA5_3_3/norm.weight
img_backbone.stage5.OSA5_3.layers.3.OSA5_3_3/norm.bias
img_backbone.stage5.OSA5_3.layers.4.OSA5_3_4/conv.weight
img_backbone.stage5.OSA5_3.layers.4.OSA5_3_4/norm.weight
img_backbone.stage5.OSA5_3.layers.4.OSA5_3_4/norm.bias
img_backbone.stage5.OSA5_3.concat.OSA5_3_concat/conv.weight
img_backbone.stage5.OSA5_3.concat.OSA5_3_concat/norm.weight
img_backbone.stage5.OSA5_3.concat.OSA5_3_concat/norm.bias
img_backbone.stage5.OSA5_3.ese.fc.weight
img_backbone.stage5.OSA5_3.ese.fc.bias
img_neck.lateral_convs.0.conv.weight
img_neck.lateral_convs.0.conv.bias
img_neck.lateral_convs.1.conv.weight
img_neck.lateral_convs.1.conv.bias
img_neck.fpn_convs.0.conv.weight
img_neck.fpn_convs.0.conv.bias
img_roi_head.cls.weight
img_roi_head.cls.bias
img_roi_head.shared_reg.0.weight
img_roi_head.shared_reg.0.bias
img_roi_head.shared_reg.1.weight
img_roi_head.shared_reg.1.bias
img_roi_head.shared_cls.0.weight
img_roi_head.shared_cls.0.bias
img_roi_head.shared_cls.1.weight
img_roi_head.shared_cls.1.bias
img_roi_head.centerness.weight
img_roi_head.centerness.bias
img_roi_head.ltrb.weight
img_roi_head.ltrb.bias
img_roi_head.center2d.weight
img_roi_head.center2d.bias
predictor.sub_graph.layers.0.linear.weight
predictor.sub_graph.layers.0.linear.bias
predictor.sub_graph.layers.0.layer_norm.weight
predictor.sub_graph.layers.0.layer_norm.bias
predictor.sub_graph.layers.1.linear.weight
predictor.sub_graph.layers.1.linear.bias
predictor.sub_graph.layers.1.layer_norm.weight
predictor.sub_graph.layers.1.layer_norm.bias
predictor.sub_graph.layers.2.linear.weight
predictor.sub_graph.layers.2.linear.bias
predictor.sub_graph.layers.2.layer_norm.weight
predictor.sub_graph.layers.2.layer_norm.bias
predictor.point_level_sub_graph.layers.0.query.weight
predictor.point_level_sub_graph.layers.0.query.bias
predictor.point_level_sub_graph.layers.0.key.weight
predictor.point_level_sub_graph.layers.0.key.bias
predictor.point_level_sub_graph.layers.0.value.weight
predictor.point_level_sub_graph.layers.0.value.bias
predictor.point_level_sub_graph.layers.1.query.weight
predictor.point_level_sub_graph.layers.1.query.bias
predictor.point_level_sub_graph.layers.1.key.weight
predictor.point_level_sub_graph.layers.1.key.bias
predictor.point_level_sub_graph.layers.1.value.weight
predictor.point_level_sub_graph.layers.1.value.bias
predictor.point_level_sub_graph.layers.2.query.weight
predictor.point_level_sub_graph.layers.2.query.bias
predictor.point_level_sub_graph.layers.2.key.weight
predictor.point_level_sub_graph.layers.2.key.bias
predictor.point_level_sub_graph.layers.2.value.weight
predictor.point_level_sub_graph.layers.2.value.bias
predictor.point_level_sub_graph.layer_0.linear.weight
predictor.point_level_sub_graph.layer_0.linear.bias
predictor.point_level_sub_graph.layer_0.layer_norm.weight
predictor.point_level_sub_graph.layer_0.layer_norm.bias
predictor.point_level_sub_graph.layers_2.0.weight
predictor.point_level_sub_graph.layers_2.0.bias
predictor.point_level_sub_graph.layers_2.1.weight
predictor.point_level_sub_graph.layers_2.1.bias
predictor.point_level_sub_graph.layers_2.2.weight
predictor.point_level_sub_graph.layers_2.2.bias
predictor.point_level_sub_graph.layers_3.0.weight
predictor.point_level_sub_graph.layers_3.0.bias
predictor.point_level_sub_graph.layers_3.1.weight
predictor.point_level_sub_graph.layers_3.1.bias
predictor.point_level_sub_graph.layers_3.2.weight
predictor.point_level_sub_graph.layers_3.2.bias
predictor.point_level_sub_graph.layers_4.0.query.weight
predictor.point_level_sub_graph.layers_4.0.query.bias
predictor.point_level_sub_graph.layers_4.0.key.weight
predictor.point_level_sub_graph.layers_4.0.key.bias
predictor.point_level_sub_graph.layers_4.0.value.weight
predictor.point_level_sub_graph.layers_4.0.value.bias
predictor.point_level_sub_graph.layers_4.1.query.weight
predictor.point_level_sub_graph.layers_4.1.query.bias
predictor.point_level_sub_graph.layers_4.1.key.weight
predictor.point_level_sub_graph.layers_4.1.key.bias
predictor.point_level_sub_graph.layers_4.1.value.weight
predictor.point_level_sub_graph.layers_4.1.value.bias
predictor.point_level_sub_graph.layers_4.2.query.weight
predictor.point_level_sub_graph.layers_4.2.query.bias
predictor.point_level_sub_graph.layers_4.2.key.weight
predictor.point_level_sub_graph.layers_4.2.key.bias
predictor.point_level_sub_graph.layers_4.2.value.weight
predictor.point_level_sub_graph.layers_4.2.value.bias
predictor.point_level_sub_graph.layer_0_again.linear.weight
predictor.point_level_sub_graph.layer_0_again.linear.bias
predictor.point_level_sub_graph.layer_0_again.layer_norm.weight
predictor.point_level_sub_graph.layer_0_again.layer_norm.bias
predictor.point_level_cross_attention.query.weight
predictor.point_level_cross_attention.query.bias
predictor.point_level_cross_attention.key.weight
predictor.point_level_cross_attention.key.bias
predictor.point_level_cross_attention.value.weight
predictor.point_level_cross_attention.value.bias
predictor.global_graph.query.weight
predictor.global_graph.query.bias
predictor.global_graph.key.weight
predictor.global_graph.key.bias
predictor.global_graph.value.weight
predictor.global_graph.value.bias
predictor.laneGCN_A2L.query.weight
predictor.laneGCN_A2L.query.bias
predictor.laneGCN_A2L.key.weight
predictor.laneGCN_A2L.key.bias
predictor.laneGCN_A2L.value.weight
predictor.laneGCN_A2L.value.bias
predictor.laneGCN_L2L.global_graph.query.weight
predictor.laneGCN_L2L.global_graph.query.bias
predictor.laneGCN_L2L.global_graph.key.weight
predictor.laneGCN_L2L.global_graph.key.bias
predictor.laneGCN_L2L.global_graph.value.weight
predictor.laneGCN_L2L.global_graph.value.bias
predictor.laneGCN_L2L.global_graph2.query.weight
predictor.laneGCN_L2L.global_graph2.query.bias
predictor.laneGCN_L2L.global_graph2.key.weight
predictor.laneGCN_L2L.global_graph2.key.bias
predictor.laneGCN_L2L.global_graph2.value.weight
predictor.laneGCN_L2L.global_graph2.value.bias
predictor.laneGCN_L2A.query.weight
predictor.laneGCN_L2A.query.bias
predictor.laneGCN_L2A.key.weight
predictor.laneGCN_L2A.key.bias
predictor.laneGCN_L2A.value.weight
predictor.laneGCN_L2A.value.bias
predictor.sub_graph_map.layers.0.linear.weight
predictor.sub_graph_map.layers.0.linear.bias
predictor.sub_graph_map.layers.0.layer_norm.weight
predictor.sub_graph_map.layers.0.layer_norm.bias
predictor.sub_graph_map.layers.1.linear.weight
predictor.sub_graph_map.layers.1.linear.bias
predictor.sub_graph_map.layers.1.layer_norm.weight
predictor.sub_graph_map.layers.1.layer_norm.bias
predictor.sub_graph_map.layers.2.linear.weight
predictor.sub_graph_map.layers.2.linear.bias
predictor.sub_graph_map.layers.2.layer_norm.weight
predictor.sub_graph_map.layers.2.layer_norm.bias
predictor.decoder.decoder.mlp.linear.weight
predictor.decoder.decoder.mlp.linear.bias
predictor.decoder.decoder.mlp.layer_norm.weight
predictor.decoder.decoder.mlp.layer_norm.bias
predictor.decoder.decoder.fc.weight
predictor.decoder.decoder.fc.bias
predictor.decoder.variety_loss_decoder.mlp.linear.weight
predictor.decoder.variety_loss_decoder.mlp.linear.bias
predictor.decoder.variety_loss_decoder.mlp.layer_norm.weight
predictor.decoder.variety_loss_decoder.mlp.layer_norm.bias
predictor.decoder.variety_loss_decoder.fc.weight
predictor.decoder.variety_loss_decoder.fc.bias
empty_linear.weight
empty_linear.bias
agents_layer_mlp_0.0.linear.weight
agents_layer_mlp_0.0.linear.bias
agents_layer_mlp_0.0.layer_norm.weight
agents_layer_mlp_0.0.layer_norm.bias
agents_layer_mlp_0.1.linear.weight
agents_layer_mlp_0.1.linear.bias
agents_layer_mlp_0.1.layer_norm.weight
agents_layer_mlp_0.1.layer_norm.bias
add_branch_mlp.linear.weight
add_branch_mlp.linear.bias
add_branch_mlp.layer_norm.weight
add_branch_mlp.layer_norm.bias
add_branch_attention.attentions.0.attn.in_proj_weight
add_branch_attention.attentions.0.attn.in_proj_bias
add_branch_attention.attentions.0.attn.out_proj.weight
add_branch_attention.attentions.0.attn.out_proj.bias
add_branch_attention.attentions.1.attn.in_proj_weight
add_branch_attention.attentions.1.attn.in_proj_bias
add_branch_attention.attentions.1.attn.out_proj.weight
add_branch_attention.attentions.1.attn.out_proj.bias
add_branch_attention.ffns.0.layers.0.0.weight
add_branch_attention.ffns.0.layers.0.0.bias
add_branch_attention.ffns.0.layers.1.weight
add_branch_attention.ffns.0.layers.1.bias
add_branch_attention.norms.0.weight
add_branch_attention.norms.0.bias
add_branch_attention.norms.1.weight
add_branch_attention.norms.1.bias
add_branch_attention.norms.2.weight
add_branch_attention.norms.2.bias