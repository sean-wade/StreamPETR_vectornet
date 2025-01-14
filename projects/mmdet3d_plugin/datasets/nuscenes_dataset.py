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
import math
import copy
import torch
import random
import numpy as np
from typing import Dict, List, Tuple

from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes

from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.eval.common.utils import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap, locations

from . import utils

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, 
                 collect_keys, 
                 seq_mode=False, 
                 seq_split_num=1, 
                 num_frame_losses=1, 
                 queue_length=8, 
                 random_length=0,
                 predict_future_frame=12,
                 predict_interval=1, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

        # for prediction
        self.predict_future_frame = predict_future_frame
        self.predict_interval = predict_interval

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)


    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            
            if not self.seq_mode: # for sliding window only
                if input_dict['scene_token'] != prev_scene_token:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict['scene_token']
                else:
                    input_dict.update(dict(prev_exists=True))

            self.pre_pipeline(input_dict)

            # for pred......
            pred_data = self.prepare_pred(i)
            input_dict.update(pred_data)
            
            example = self.pipeline(input_dict)
            queue.append(example)

        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and \
                (queue[-k-1] is None or ~(queue[-k-1]['gt_labels_3d']._data != -1).any()):
                return None
        return self.union2one(queue)

    
    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)

        # for pred......
        pred_data = self.prepare_pred(index)
        input_dict.update(pred_data)
        
        example = self.pipeline(input_dict)
        return example
        
    
    def union2one(self, queue):
        for key in self.collect_keys:
            # if key != 'img_metas':
            if key not in ['img_metas', 'pred_mapping', 'pred_polyline_spans', 'pred_matrix', 'instance_idx_2_labels']:
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
        if not self.test_mode:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths']:
                if key == 'gt_bboxes_3d':
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                else:
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)

        queue = queue[-1]
        return queue

    
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        e2g_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_translation = info['ego2global_translation']
        l2e_rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_translation = info['lidar2ego_translation']
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        ego_pose =  e2g_matrix @ l2e_matrix # lidar2global

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                cam2lidar_r = cam_info['sensor2lidar_rotation']
                cam2lidar_t = cam_info['sensor2lidar_translation']
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
                
            if not self.test_mode: # for seq_mode
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                ))
        # if not self.test_mode:
        if 1:
            annos = self.get_ann_info(index)
            annos.update( 
                dict(
                    bboxes=info['bboxes2d'],
                    labels=info['labels2d'],
                    centers2d=info['centers2d'],
                    depths=info['depths'],
                    bboxes_ignore=info['bboxes_ignore'])
            )
            input_dict['ann_info'] = annos
            
        return input_dict


    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        instance_inds = np.array(info['instance_inds'], dtype=np.int)[mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            instance_inds=instance_inds)
        return anns_results
    
    
    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    
    ###############################################################################
    ################################## for prediction #############################
    def prepare_pred(self, index):
        results = {}
        if not hasattr(self, 'nuscenes'):
            self.prepare_nuscenes()

        # judge same scene.
        same_scene = self.is_the_same_scene(index, self.predict_future_frame)

        mapping = {}

        self.get_pred_agents(results, index, mapping)
        self.get_pred_lanes(results, index, mapping)

        info = self.data_infos[index]
        sample = self.helper.data.get('sample', info['token'])
        scene_id = self.helper.data.get('scene', sample['scene_token'])['name']

        mapping.update(dict(
            timestamp=info['timestamp'] / 1e6,
            timestamp_origin=info['timestamp'],
            sample_token=info['token'],
            scene_id=scene_id,
            same_scene=same_scene,
            index=index,
        ))

        # if hasattr(self, 'work_dir'):
        #     mapping['work_dir'] = self.work_dir

        results['pred_mapping'] = mapping
        return results
    

    def prepare_nuscenes(self):
        # self.nuscenes = NuScenes('v1.0-trainval/', dataroot=self.data_root)
        self.nuscenes = NuScenes('v1.0-mini', dataroot=self.data_root)
        self.helper = PredictHelper(self.nuscenes)
        self.maps = load_all_maps(self.helper)


    def is_the_same_scene(self, start, future_frame_num):
        timestamps = []
        for i in range(start, future_frame_num):
            if i < 0 or i >= len(self.data_infos):
                return False
            info = self.data_infos[i]
            timestamp = info['timestamp'] / 1e6
            timestamps.append(timestamp)

        for i in range(len(timestamps) - 1):
            if abs(timestamps[i + 1] - timestamps[i]) > 10:
                self.last_index = start + i
                return False

        return True


    def get_transform_and_rotate(self, point, translation, rotation, reverse=False):
        if reverse:
            quaternion = Quaternion(rotation).inverse
            point = point - translation
            point = np.dot(quaternion.rotation_matrix, point)
        else:
            quaternion = Quaternion(rotation)
            point = np.dot(quaternion.rotation_matrix, point)
            point = point + translation
        return point
    

    def get_pred_lanes(self, results, index, mapping):
        cur_info = self.data_infos[index]
        cur_l2e_r = cur_info['lidar2ego_rotation']
        cur_l2e_t = cur_info['lidar2ego_translation']
        cur_e2g_r = cur_info['ego2global_rotation']
        cur_e2g_t = cur_info['ego2global_translation']

        point = np.array([0.0, 0.0, 0.0])
        point = self.get_transform_and_rotate(point, cur_e2g_t, cur_e2g_r)
        agent_x, agent_y = point[0], point[1]   # ego-av coord in global

        sample_token = cur_info['token']
        map_name = self.helper.get_map_name_from_sample_token(sample_token)

        #### Attention !!! Add by zh: 替换成自车yaw角，且使用(30,0)为基准点
        yaw = Quaternion(cur_e2g_r).yaw_pitch_roll[0]
        normalizer = utils.Normalizer(agent_x, agent_y, -yaw)

        max_dis = 70.0
        visible_x = 30.0
        discretization_resolution_meters = 1
        nuscene_lanes = get_lanes_in_radius(agent_x, agent_y, max_dis, discretization_resolution_meters, self.maps[map_name])

        vectors = []
        polygons = []
        polyline_spans = []

        def get_dis_point2point(point, point_=(0.0, 0.0)):
            return np.sqrt(np.square((point[0] - point_[0])) + np.square((point[1] - point_[1])))

        if True:
            for poses_along_lane in nuscene_lanes.values():
                lane = [pose[:2] for pose in poses_along_lane]
                lane = np.array(lane)
                lane = normalizer(lane)     # convert to ego coord.

                lane = np.array([point for point in lane if get_dis_point2point(point, (visible_x, 0.0)) < max_dis])
                if len(lane) < 1:
                    continue

                polygons.append(lane)

                start = len(vectors)
                stride = 5
                scale = 0.05
                vector = np.zeros(128)
                for j in range(0, len(lane), stride):
                    cur = 0
                    for k in range(stride + 2):
                        t = min(j + k, len(lane) - 1)
                        vector[cur + 2 * k + 0] = lane[t, 0] * scale
                        vector[cur + 2 * k + 1] = lane[t, 1] * scale

                    # cur = 30
                    # if type[now] != -1:
                    #     assert type[now] < 20
                    #     vector[cur + type[now]] = 1.0

                    cur = 40
                    vector[cur + 0] = j
                    t_float = j
                    vector[cur + 1] = t_float / len(lane)

                    vectors.append(copy.copy(vector))  # Attention!!! ViP3D repo has a huge bug here!!!

                if len(vectors) > start:
                    polyline_spans.append(slice(start, len(vectors)))

        if len(polyline_spans) == 0:
            start = len(vectors)
            assert start == 0

            vectors.append(np.zeros(128))
            polyline_spans.append(slice(start, len(vectors)))

            lane = np.zeros([1, 2], dtype=np.float32)
            polygons.append(lane)

        mapping.update(dict(
            lanes=polygons,
            map_name=map_name,
        ))

        results.update(dict(
            pred_matrix=np.array(vectors, dtype=np.float32),
            pred_polyline_spans=polyline_spans,
        ))


    def get_future_pipeline(self):
        from mmdet3d.datasets.pipelines import Compose
        pipeline = [
                dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
                                            with_label=True, with_bbox_depth=True),
                dict(type='InstanceRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(type='ObjectNameFilter', classes=[
                        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
                    ),
            ]
        return Compose(pipeline)


    def prepare_data_future(self, start, end, interval):
        ret = None
        for i in range(start, end, interval):
            if not (0 <= i < len(self.data_infos)):
                return None
            
            if not self.test_mode:
                input_dict = self.get_data_info(i)
                input_dict.update(dict(prev_exists=True))
                
                self.pre_pipeline(input_dict)
                pipeline = self.get_future_pipeline()
                data_i = pipeline(input_dict)
                data_i['instance_inds'] = data_i['ann_info']['instance_inds']
            else:
                input_dict = self.get_data_info(i)
                data_i = {}
                data_i['gt_bboxes_3d'] = input_dict['ann_info']['gt_bboxes_3d']
                data_i['gt_labels_3d'] = input_dict['ann_info']['gt_labels_3d']
                data_i['instance_inds'] = input_dict['ann_info']['instance_inds']

            if ret is None:
                ret = {key: [] for key in data_i.keys()}

            for key, value in data_i.items():
                ret[key].append(value)
        return ret


    def get_pred_agents(self, results, index, mapping):
        future_frame_num = self.predict_future_frame
        instance_idx_2_labels = {}

        cur_info = self.data_infos[index]
        cur_l2e_r = cur_info['lidar2ego_rotation']
        cur_l2e_t = cur_info['lidar2ego_translation']
        cur_e2g_r = cur_info['ego2global_rotation']
        cur_e2g_t = cur_info['ego2global_translation']

        same_scene = self.is_the_same_scene(index, future_frame_num)

        data_future = self.prepare_data_future(index, index + future_frame_num, self.predict_interval)

        if same_scene and data_future is not None:
            instance_inds = data_future['instance_inds'][0]
            # cur index exists ids
            gt_bboxes_3d_cur = data_future['gt_bboxes_3d'][0].tensor.numpy()
            for instance_idx in instance_inds:
                assert instance_idx != -1
                if instance_idx not in instance_idx_2_labels:
                    instance_idx_2_labels[instance_idx] = dict(
                        future_traj=np.zeros((future_frame_num, 2), dtype=np.float32),
                        future_traj_relative=np.zeros((future_frame_num, 2), dtype=np.float32),
                        future_traj_is_valid=np.zeros(future_frame_num, dtype=np.int32),
                        future_yaw = np.zeros(future_frame_num, dtype=np.float32),
                        # TODO: get past trajs...
                        # past_traj=np.zeros((past_frame_num, 2), dtype=np.float32),
                        # past_traj_is_valid=np.zeros(past_frame_num, dtype=np.int32),
                        # category=np.zeros(past_frame_num, dtype=np.int32),
                        # past_boxes=np.zeros((past_frame_num, 7), dtype=np.float32),
                    )
        
            for i in range(index, index + future_frame_num):
                info = self.data_infos[i]
                l2e_r = info['lidar2ego_rotation']
                l2e_t = info['lidar2ego_translation']
                e2g_r = info['ego2global_rotation']
                e2g_t = info['ego2global_translation']

                instance_inds = data_future['instance_inds'][i - index]
                gt_bboxes_3d = data_future['gt_bboxes_3d'][i - index].tensor.numpy()
                gt_labels_3d = data_future['gt_labels_3d'][i - index]
                 
                for box_idx, instance_idx in enumerate(instance_inds):
                    if instance_idx not in instance_idx_2_labels:
                        continue

                    box = utils.get_box_from_array(gt_bboxes_3d[box_idx])
                    box = utils.get_transform_and_rotate_box(box, l2e_t, l2e_r)     # t_lidar -> t_ego
                    box = utils.get_transform_and_rotate_box(box, e2g_t, e2g_r)     # t_ego -> global
                    box = utils.get_transform_and_rotate_box(box, cur_e2g_t, cur_e2g_r, reverse=True)   #  global -> ego_t0
                    box = utils.get_transform_and_rotate_box(box, cur_l2e_t, cur_l2e_r, reverse=True)   #  ego_t0 -> lidar_t0
                    point = box.center
                    # yaw = -(gt_bboxes_3d[box_idx][6] + np.pi / 2)
                    
                    instance_idx_2_labels[instance_idx]['future_traj'][i - index] = point[0], point[1]
                    instance_idx_2_labels[instance_idx]['future_yaw'][i - index] = -(gt_bboxes_3d[box_idx][6] + np.pi / 2)
                    x, y = instance_idx_2_labels[instance_idx]['future_traj'][0][0], \
                           instance_idx_2_labels[instance_idx]['future_traj'][0][1]
                    yaw = instance_idx_2_labels[instance_idx]['future_yaw'][0]
                    normalizer = utils.Normalizer(x, y, -yaw)
                    instance_idx_2_labels[instance_idx]['future_traj_relative'][i - index] = normalizer(point[:2])
                    instance_idx_2_labels[instance_idx]['future_traj_is_valid'][i - index] = 1

        instance_inds = None
        if same_scene and data_future is not None:
            instance_inds = data_future['instance_inds'][0]

        mapping.update(dict(
            cur_l2e_r=cur_l2e_r,
            cur_l2e_t=cur_l2e_t,
            cur_e2g_r=cur_e2g_r,
            cur_e2g_t=cur_e2g_t,
            # r_index_2_rotation_and_transform=r_index_2_rotation_and_transform,
            valid_pred=same_scene and data_future is not None,
            instance_inds=instance_inds,
        ))

        results['instance_idx_2_labels'] = instance_idx_2_labels
    
    ###############################################################################


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


###############################################################################
################################## for prediction #############################

def load_all_maps(helper: PredictHelper, verbose: bool = False) -> Dict[str, NuScenesMap]:
    """
    Loads all NuScenesMap instances for all available maps.
    :param helper: Instance of PredictHelper.
    :param verbose: Whether to print to stdout.
    :return: Mapping from map-name to the NuScenesMap api instance.
    """
    dataroot = helper.data.dataroot
    maps = {}

    for map_name in locations:
        if verbose:
            print(f'static_layers.py - Loading Map: {map_name}')

        maps[map_name] = NuScenesMap(dataroot, map_name=map_name)

    return maps


def get_lanes_in_radius(x: float, y: float, radius: float,
                        discretization_meters: float,
                        map_api: NuScenesMap) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Retrieves all the lanes and lane connectors in a radius of the query point.
    :param x: x-coordinate of point in global coordinates.
    :param y: y-coordinate of point in global coordinates.
    :param radius: Any lanes within radius meters of the (x, y) point will be returned.
    :param discretization_meters: How finely to discretize the lane. If 1 is given, for example,
        the lane will be discretized into a list of points such that the distances between points
        is approximately 1 meter.
    :param map_api: The NuScenesMap instance to query.
    :return: Mapping from lane id to list of coordinate tuples in global coordinate system.
    """

    lanes = map_api.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
    lanes = lanes['lane'] + lanes['lane_connector']
    lanes = map_api.discretize_lanes(lanes, discretization_meters)

    return lanes
###############################################################################