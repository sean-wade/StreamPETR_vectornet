import os
import torch
import importlib
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mmcv import Config
from mmdet3d.datasets import build_dataset


def viz_train_data_item(data, save_path=None):
    _, ax = plt.subplots(figsize=(12, 12))
    ax.axis('equal')
    # plt.axis('off')
    ax.set_title('seq_id: {}'.format(data['pred_mapping'].data[0]['sample_token']))
    
    # lane
    for lane in data['pred_mapping'].data[0]['lanes']:
        # TODO: check whether to draw arrows.
        ax.plot(lane[:, 0], lane[:, 1], marker='.', alpha=0.5, color="grey")

    # past
    i = 0
    for past_traj, past_traj_valid, future_traj, future_traj_valid in \
            zip(data['past_traj'].data[0], data['past_traj_is_valid'].data[0], data['future_traj'].data[0], data['future_traj_is_valid'].data[0]):
        past_traj = past_traj[past_traj_valid.numpy().astype(np.bool)]
        future_traj = future_traj[future_traj_valid.numpy().astype(np.bool)]
        color = tuple(np.random.rand(3,))
        # past
        ax.plot(past_traj[:, 0], past_traj[:, 1], marker='.', alpha=0.5, color=color, zorder=15)
        # orig
        ax.plot(past_traj[-1, 0], past_traj[-1, 1], alpha=0.5, color=color, marker='o', zorder=5, markersize=10)
        # future
        ax.plot(future_traj[:, 0], future_traj[:, 1], marker='.', alpha=0.5, color=color, zorder=15)
        i+=1

    if save_path:
        plt.savefig(save_path + "/" + data['pred_mapping'].data[0]['sample_token'] + ".png")
    else:
        plt.show()
    plt.close()


def viz_test_data_item(data, save_path=None):
    _, ax = plt.subplots(figsize=(12, 12))
    ax.axis('equal')
    # plt.axis('off')
    ax.set_title('seq_id: {}'.format(data['pred_mapping'][0].data['sample_token']))
    
    # lane
    for lane in data['pred_mapping'][0].data['lanes']:
        # TODO: check whether to draw arrows.
        ax.plot(lane[:, 0], lane[:, 1], marker='.', alpha=0.5, color="grey")

    # past
    i = 0
    for past_traj, past_traj_valid, future_traj, future_traj_valid in \
            zip(data['past_traj'][0].data, data['past_traj_is_valid'][0].data, data['future_traj'][0].data, data['future_traj_is_valid'][0].data):
        past_traj = past_traj[past_traj_valid.numpy().astype(np.bool)]
        future_traj = future_traj[future_traj_valid.numpy().astype(np.bool)]
        color = tuple(np.random.rand(3,))
        # past
        ax.plot(past_traj[:, 0], past_traj[:, 1], marker='.', alpha=0.5, color=color, zorder=15)
        # orig
        ax.plot(past_traj[-1, 0], past_traj[-1, 1], alpha=0.5, color=color, marker='o', zorder=5, markersize=10)
        # future
        ax.plot(future_traj[:, 0], future_traj[:, 1], marker='.', alpha=0.5, color=color, zorder=15)
        i+=1

    if save_path:
        plt.savefig(save_path + "/" + data['pred_mapping'][0].data['sample_token'] + ".png")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    from tqdm import tqdm
    cfg = Config.fromfile("projects/configs/vectornet/stream_petr_vov_flash_800_bs16_wk4_seq_24e_mini.py")

    plugin_dir = cfg.plugin_dir
    _module_dir = os.path.dirname(plugin_dir)
    _module_dir = _module_dir.split('/')
    _module_path = _module_dir[0]
    for m in _module_dir[1:]:
        _module_path = _module_path + '.' + m
    print(_module_path)
    plg_lib = importlib.import_module(_module_path)

    TEST = True
    if TEST:
        dataset = build_dataset(cfg.data.test)
        save_path = "work_dirs/viz_test"
    else:
        dataset = build_dataset(cfg.data.train)
        save_path = "work_dirs/viz_train"

    os.makedirs(save_path, exist_ok=True)

    for dd in tqdm(dataset):
        if TEST:
            viz_test_data_item(dd, save_path)
        else:
            viz_train_data_item(dd, save_path)
