# coding:utf-8
import os
from collections.abc import Sequence

import cv2
import torch
from torch.nn import functional as F
from torchvision.ops import nms
import numpy as np


class DecodeBox():
    def __init__(
            self, 
            std: torch.Tensor, 
            num_classes: int
        ):
        self.std            = std
        self.num_classes    = num_classes + 1    


    def frcnn_correct_boxes(
            self, 
            box_xy, 
            box_wh, 
            input_shape, 
            image_shape
        ):
        # 把 y 轴放前面是因为方便预测框和图像的宽高进行相乘
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes


    def forward(
            self, 
            roi_cls_locs: torch.Tensor, 
            roi_scores: torch.Tensor, 
            rois: torch.Tensor, 
            image_shape: Sequence, 
            input_shape: Sequence, 
            nms_iou: float = 0.3, 
            confidence: float = 0.5
        ):
        results = []
        bs = len(roi_cls_locs)
        # batch_size, num_rois, 4
        rois    = rois.view((bs, -1, 4))
        # 在 predict 的时候只输入一张图片，所以 for i in range(len(mbox_loc)) 只进行一次
        for i in range(bs):
            # 对回归参数进行reshape
            roi_cls_loc = roi_cls_locs[i] * self.std
            # 第一维度是建议框的数量，第二维度是每个种类
            # 第三维度是对应种类的调整参数
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])

            # 利用 classifier 网络的预测结果对建议框进行调整获得预测框
            # num_rois, 4 -> num_rois, 1, 4 -> num_rois, num_classes, 4
            roi = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            cls_bbox = cls_bbox.view([-1, (self.num_classes), 4])
            # 对预测框进行归一化，调整到 0-1 之间
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

            roi_score = roi_scores[i]
            prob = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                # 取出属于该类的所有框的置信度
                # 判断是否大于阈值
                c_confs = prob[:, c]
                c_confs_m = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    # 取出得分高于confidence的框
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )
                    # 取出在非极大抑制中效果较好的内容
                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))
                    # 将label、置信度、框的位置进行堆叠
                    c_pred = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    # 添加进 result 里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        return results


def generate_anchor_base(
        base_size: int = 16, 
        ratios: list = [0.5, 1, 2], 
        anchor_scales: list = [8, 16, 32]
    ):
    """生成基础的先验框"""
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


def enumerate_shifted_anchor(
        anchor_base: np.ndarray, 
        feat_stride: int,
        height: int,
        width: int
    ):
    """对基础先验框进行拓展对应到所有特征点上"""
    # 计算网格中心点
    shift_x          = np.arange(0, width * feat_stride, feat_stride)
    shift_y          = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift            = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    # 所有的先验框
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def loc2bbox(src_bbox: torch.Tensor, loc: torch.Tensor):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width  = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x  = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y  = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox


def get_classes(classes_path: str | os.PathLike):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_new_image_size(height: int, width: int, img_min_side: int = 600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width


def resize_image(image: np.ndarray, size: Sequence):
    image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return image
