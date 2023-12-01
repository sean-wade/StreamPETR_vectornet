# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.core.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

# from tools.onnx_utils import nuscenceData, PetrWrapper, get_onnx_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
# from matplotlib.axes import Axes
# from typing import Tuple, List, Dict
# from nuscenes.utils.geometry_utils import view_points, transform_matrix
from tools.data_converter.waymoKitti_dataset import kitti_label_cv2, kitti_label_plt

from visual_utils import draw_box2d, Box

label_to_name = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config',help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    print("Annotation:{} total {} samples".format(cfg.data.val.ann_file, len(dataset)))

    # load result json
    i_ckpt = 0
    result_paths = ['test/stream_petr_r50_704x256_seq_428q_nui_24e_waymo/Mon_Oct_30_10_21_46_2023/results_waymokitti.json']

    results = mmcv.load(result_paths[i_ckpt])


    def visualize_sample(idx, outpath):
        data = dataset[idx]
        print(data.keys())
        for key in data.keys():
            print("{}:{}".format(key, data[key]))
        imgs = data['img'][0].data
        lidar2imgs = data['lidar2img'][0].data

        pred_bboxes_3d = np.array(results[idx]['pts_bbox']['boxes_3d'])
        pred_scores_3d = np.array(results[idx]['pts_bbox']['scores_3d'])
        pred_labels_3d = np.array(results[idx]['pts_bbox']['labels_3d'])

        # filter confidence
        conf_th = 0.3
        mask = [i for i, score in enumerate(pred_scores_3d) if score > conf_th]

        pred_bboxes_3d = pred_bboxes_3d[mask]
        pred_labels_3d = pred_labels_3d[mask]
        pred_scores_3d = pred_scores_3d[mask]


        # 3d box
        print("Showing {} 3d box & bev...".format(cfg.dataset_type))
        # waymo and nusc has different camera number
        plt.rcParams['figure.figsize'] = (16, 9)
        if cfg.dataset_type == 'CustomWaymoKittiDataset':
            fig, axes = plt.subplots(2, 3)
            camid_to_ax = {0: [0, 1], 1: [0, 0], 2: [0, 2], 3: [1, 0], 4: [1, 2]}
            bevid = [1, 1]
        elif cfg.dataset_type == 'CustomNuScenesDataset':
            fig, axes = plt.subplots(3, 3)
            camid_to_ax = {0: [0, 1], 1: [0, 2], 2: [0, 0], 3: [1, 1], 4: [1, 0], 5: [1, 2]}
            bevid = [2, 1]
            axes[2][0].axis('off')
            axes[2][2].axis('off')
        else:
            raise ValueError('Not implement...')
        # plot box in img
        for i, img in enumerate(imgs):
            img = np.transpose(img.numpy(), (1,2,0)).astype(np.uint8)   # disable NormalizeMultiviewImage
            lidar2img = lidar2imgs[i].numpy()
            for ith, box in enumerate(pred_bboxes_3d):
                box = Box(box.tolist())
                img = box.draw_2d_3dbox(img.copy(), view=lidar2img, colors=(kitti_label_cv2[label_to_name[pred_labels_3d[ith]]], kitti_label_cv2[label_to_name[pred_labels_3d[ith]]], kitti_label_cv2[label_to_name[pred_labels_3d[ith]]]))
            img = img[..., ::-1]
            axes[camid_to_ax[i][0]][camid_to_ax[i][1]].imshow(Image.fromarray(img))
            axes[camid_to_ax[i][0]][camid_to_ax[i][1]].axis('off')
        # plot bev
        print(pred_bboxes_3d)
        axes[bevid[0]][bevid[1]].plot(0, 0, 'x', color='red')
        # plot distance circle
        circle30 = patches.Circle((0, 0), radius=30, linewidth=1, linestyle='dashed', fill=False)
        circle50 = patches.Circle((0, 0), radius=50, linewidth=1, linestyle='dashed', fill=False)
        circle70 = patches.Circle((0, 0), radius=70, linewidth=1, linestyle='dashed', fill=False)
        axes[bevid[0]][bevid[1]].add_patch(circle30)
        axes[bevid[0]][bevid[1]].text(30, 0, '30m', ha='center', va='center')
        axes[bevid[0]][bevid[1]].add_patch(circle50)
        axes[bevid[0]][bevid[1]].text(50, 0, '50m', ha='center', va='center')
        axes[bevid[0]][bevid[1]].add_patch(circle70)
        axes[bevid[0]][bevid[1]].text(70, 0, '70m', ha='center', va='center')

        for ith,box in enumerate(pred_bboxes_3d):
            # nusc box: x,y,z,l,w,h,rot,vx,vy
            if len(box.tolist()) == 9:
                view_mat = np.eye(4)
            # waymokitti box: x,y,z,l,w,h,rot
            elif len(box.tolist()) == 7:
                view_mat = np.array([0., -1., 1., 0.]).reshape(2, 2)
            else:
                raise ValueError('bbox3d not supported yet')
            box = Box(box.tolist())
            box.render(axes[bevid[0]][bevid[1]], pred_scores_3d[ith],view=view_mat, colors=(kitti_label_plt[label_to_name[pred_labels_3d[ith]]], kitti_label_plt[label_to_name[pred_labels_3d[ith]]], kitti_label_plt[label_to_name[pred_labels_3d[ith]]]))
        axes[bevid[0]][bevid[1]].set_xlim((cfg.point_cloud_range[0], cfg.point_cloud_range[3]))
        axes[bevid[0]][bevid[1]].set_ylim((cfg.point_cloud_range[1], cfg.point_cloud_range[4]))
        print("\n")

        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        # plt.savefig("result_waymo/pred_{}.jpg".format(str(idx).zfill(7)), dpi=500, bbox_inches='tight')
        plt.savefig(outpath, dpi=500, bbox_inches='tight')

    indices = [i for i in range(0,100,5)]
    for idx in indices:
        outpath = "result_waymo/pred_{}_{}th.jpg".format(str(idx).zfill(7), i_ckpt)
        visualize_sample(idx, outpath)



    # plt.show()
    # plt.close()

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
