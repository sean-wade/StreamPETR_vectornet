import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .visual_Box import draw_box2d, Box
import mmcv

# 'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
label_to_name = {0: 'Car',
                 1: 'Car',
                 2: 'Car',
                 3: 'Car',
                 4: 'Car',
                 5: 'Barrier',
                 6: 'Cyclist',
                 7: 'Cyclist',
                 8: 'Pedestrian',
                 9: 'Traffic_cone',
                 }


label_color_label = {'Car': 'g', 'Pedestrian': 'r', 'Cyclist': 'orange', 'Barrier' : 'grey', 'Traffic_cone' : 'purple'}
label_color_pixel = {'Car': (0, 255, 0), 'Pedestrian': (0, 0, 255), 'Cyclist': (0, 183, 235), 'Barrier' : (183, 183, 235), 'Traffic_cone' : (235, 183, 0)}
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


def visualize_sample(data, outpath, result):
    imgs = data['img'][0].data[0][0] #[6,3,320,800]
    lidar2imgs = data['lidar2img'][0].data[0][0]  #[6,4,4]

    pred_bboxes_3d = result[0]['pts_bbox']['boxes_3d']
    pred_scores_3d = np.array(result[0]['pts_bbox']['scores_3d'])
    pred_labels_3d = np.array(result[0]['pts_bbox']['labels_3d'])

    conf_th = 0.3
    mask = [i for i, score in enumerate(pred_scores_3d) if score > conf_th]
    pred_bboxes_3d = pred_bboxes_3d[mask]
    pred_labels_3d = pred_labels_3d[mask]
    pred_scores_3d = pred_scores_3d[mask]

    # waymo and nusc has different camera number
    plt.rcParams['figure.figsize'] = (24, 12)
    fig, axes = plt.subplots(3, 3)
    camid_to_ax = {0: [0, 1], 1: [0, 2], 2: [0, 0], 3: [1, 1], 4: [1, 0], 5: [1, 2]}
    bevid = [2, 0]
    prediction_gt_id = [2,1]
    prediction_id = [2,2]
    axes[2][1].axis('off')
    axes[2][2].axis('off')

    # plot box in img
    for i, img in enumerate(imgs):
        img = np.transpose(img.numpy(), (1,2,0))
        img = mmcv.imdenormalize(img, np.array([103.530, 116.280, 123.675]), 
                                 np.array([57.375, 57.120, 58.395]), to_bgr=False).astype(np.uint8)
        lidar2img = lidar2imgs[i].numpy()
        for ith, box in enumerate(pred_bboxes_3d):
            box = Box(box.tolist())
            img = box.draw_2d_3dbox(img.copy(), view=lidar2img, colors=(label_color_pixel[label_to_name[pred_labels_3d[ith]]], label_color_pixel[label_to_name[pred_labels_3d[ith]]], label_color_pixel[label_to_name[pred_labels_3d[ith]]]))
        img = img[..., ::-1]
        axes[camid_to_ax[i][0]][camid_to_ax[i][1]].imshow(Image.fromarray(img))
        axes[camid_to_ax[i][0]][camid_to_ax[i][1]].axis('off')

    # plot bev
    ax_bev = axes[bevid[0]][bevid[1]]
    ax_bev.plot(0, 0, 'x', color='red')
    # plot distance circle
    circle30 = patches.Circle((0, 0), radius=30, linewidth=1, linestyle='dashed', fill=False)
    circle50 = patches.Circle((0, 0), radius=50, linewidth=1, linestyle='dashed', fill=False)
    circle70 = patches.Circle((0, 0), radius=70, linewidth=1, linestyle='dashed', fill=False)
    ax_bev.add_patch(circle30)
    ax_bev.text(30, 0, '30m', ha='center', va='center')
    ax_bev.add_patch(circle50)
    ax_bev.text(50, 0, '50m', ha='center', va='center')
    ax_bev.add_patch(circle70)
    ax_bev.text(70, 0, '70m', ha='center', va='center')

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
        box.render(ax_bev, 
                   pred_scores_3d[ith],
                   view=view_mat, 
                   colors=(label_color_label[label_to_name[pred_labels_3d[ith]]], label_color_label[label_to_name[pred_labels_3d[ith]]], label_color_label[label_to_name[pred_labels_3d[ith]]])
                   )
    ax_bev.set_xlim((point_cloud_range[0], point_cloud_range[3]))
    ax_bev.set_ylim((point_cloud_range[1], point_cloud_range[4]))

    if 'pred_outputs' in result[0]['pts_bbox']:
        draw_lane_prediction(data, result, axes[prediction_gt_id[0]][prediction_gt_id[1]], axes[prediction_id[0]][prediction_id[1]])
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    file_path = os.path.join(outpath, str(data['timestamp'][0].data[0][0].item())+".jpg")
    plt.savefig(file_path, dpi=500, bbox_inches='tight')
    plt.close()


def draw_lane_prediction(data, pred_result, ax_gt, ax_pred):
    ax_gt.axis('equal')
    # plt.axis('off')
    ax_gt.set_title('seq_id: {}'.format(data['pred_mapping'][0].data[0][0]['sample_token']))
    ax_pred.axis('equal')
    ax_pred.set_title('seq_id: {}'.format(data['pred_mapping'][0].data[0][0]['sample_token']))

    # lane
    for lane in data['pred_mapping'][0].data[0][0]['lanes']:
        # TODO: check whether to draw arrows.
        ax_gt.plot(lane[:, 0], lane[:, 1], marker='.', alpha=0.1, color="grey", linewidth=0.2, markersize=10)
        ax_pred.plot(lane[:, 0], lane[:, 1], marker='.', alpha=0.1, color="grey", linewidth=0.2, markersize=10)

    if 'past_traj' in data:
        # traj
        # lidar orin.
        ax_gt.plot([0,0], marker='o', markersize=5, color='r')

        for past_traj, past_traj_valid, future_traj, future_traj_valid in \
                zip(data['past_traj'][0].data[0][0], data['past_traj_is_valid'][0].data[0][0], data['future_traj'][0].data[0][0], data['future_traj_is_valid'][0].data[0][0]):
            past_traj = past_traj[past_traj_valid.bool()]
            future_traj = future_traj[future_traj_valid.bool()]
            color = tuple(np.random.rand(3,))
            # past
            ax_gt.plot(past_traj[:, 0], past_traj[:, 1], marker='.', alpha=0.5, color=color, linewidth=0.2, zorder=15, markersize=1)
            # orig
            ax_gt.plot(past_traj[-1, 0], past_traj[-1, 1], alpha=0.3, color=color, marker='o', zorder=5, markersize=3)
            if len(future_traj) > 0:
                ax_gt.plot([past_traj[-1, 0], future_traj[0, 0]], [past_traj[-1, 1], future_traj[0, 1]], color=color, linewidth=0.4)
                # future
                # ax_gt.plot(future_traj[:, 0], future_traj[:, 1], marker='.', alpha=1, color=(1, 0., 0.), linewidth=0.2, zorder=15, markersize=1)
                ax_gt.plot(future_traj[:, 0], future_traj[:, 1], marker='.', alpha=1, color=color, linewidth=0.4, zorder=15, markersize=1)
                ax_gt.plot(future_traj[-1, 0], future_traj[-1, 1], alpha=0.8, color=color, marker='^', zorder=5, markersize=2)

    if "pts_bbox" in pred_result[0]:
        ax_pred.plot([0,0], marker='o', markersize=5, color='r')

        for i in range(len(pred_result[0]['pts_bbox']['pred_outputs'])):
            max_idx = np.argmax(pred_result[0]['pts_bbox']['pred_probs'][i])
            future_traj = pred_result[0]['pts_bbox']['pred_outputs'][i][max_idx]
            orig = pred_result[0]['pts_bbox']['boxes_3d']
            color = tuple(np.random.rand(3,))

            # orig
            ax_pred.plot(orig.center[i, 0], orig.center[i, 1], alpha=0.5, color=color, marker='o', zorder=5, markersize=2)
            # future
            ax_pred.plot(future_traj[:, 0], future_traj[:, 1], marker='.', alpha=1, color=color, linewidth=0.2, zorder=15, markersize=1)
            ax_pred.plot(future_traj[-1, 0], future_traj[-1, 1], alpha=0.8, color=color, marker='^', zorder=5, markersize=2)