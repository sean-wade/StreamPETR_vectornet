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