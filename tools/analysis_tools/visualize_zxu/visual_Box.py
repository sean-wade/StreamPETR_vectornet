import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.axes import Axes
from typing import Tuple, List, Dict
from nuscenes.utils.geometry_utils import view_points, transform_matrix


def draw_box2d(img, gt_bboxes):
    img = np.transpose(img.numpy(), (1,2,0))
    img = img.astype(np.uint8)
    for box in gt_bboxes:
        print("W={},H={}".format(box[2]-box[0], box[3]-box[1]))
        img = cv2.rectangle(img.copy(), (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 1)
    return img


class Box:
    """
    nusc lidar coordinate(bev):
                y+
                 |
                 |
                 |
            z+ up.------- x+
    waymo lidar coordinate(bev):
                 x+
                 |
                 |
                 |
       y+ -------. up z+
    """
    def __init__(self, gt_bbox):
        x, y, z, l, w, h, yaw = None, None, None, None, None, None, None
        self.dataset_type = 'nusc'
        if len(gt_bbox) == 7:       # waymokitti
            x, y, z, l, w, h, yaw = gt_bbox
            self.dataset_type = 'waymo'
        elif len(gt_bbox) == 9:     # nuscenes
            x, y, z, l, w, h, yaw, vx, vy = gt_bbox
        else:
            raise ValueError('not implement yet')
        self.center = np.array([x, y, z])
        self.lwh = np.array([l, w, h])
        self.yaw = yaw  # yaw angle around z-axis

    def rot_mat(self):
        R = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0],
                      [np.sin(self.yaw), np.cos(self.yaw), 0],
                      [0, 0, 1]], dtype=np.float32)
        return R

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        l, w, h = self.lwh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        if self.dataset_type != 'nusc':
            z_corners = (z_corners * 2 + h) / 2
        else:
            z_corners += h/2
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.rot_mat(), corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def render(self,
               axis: Axes,
               conf: float = 1.0,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 1) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)

        # show confidence
        self.yaw = self.yaw % (2*np.pi)
        if self.yaw >= 3/2*np.pi or self.yaw <= 1/2*np.pi:
            axis.text(corners.T[0][0]-1, corners.T[0][1]+1, "{:.2f}".format(conf), fontsize=3, color=colors[0],
                      ha='center', va='center')
        else:
            axis.text(corners.T[0][0]+1, corners.T[0][1]-1, "{:.2f}".format(conf), fontsize=3, color=colors[0],
                      ha='center', va='center')
    def draw_2d_3dbox(self,
               img,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)

        # skip box not in this camera
        if (corners[2, :] < 0).any():
            return img

        corners[0, :] /= corners[2, :]
        corners[1, :] /= corners[2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(img,
                         [int(prev[0]), int(prev[1])],
                         [int(corner[0]), int(corner[1])],
                         color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(img,
                    [int(corners.T[i][0]), int(corners.T[i][1])],
                    [int(corners.T[i+4][0]), int(corners.T[i + 4][1])],
                    colors[2], linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(img,
                [int(center_bottom[0]), int(center_bottom[1])],
                [int(center_bottom_forward[0]), int(center_bottom_forward[1])],
                (255, 0, 255), linewidth)
        cv2.line(img,
                 [int(corners.T[2][0]), int(corners.T[2][1])],
                 [int(corners.T[3][0]), int(corners.T[3][1])],
                 (255, 0, 255), linewidth)
        return img


