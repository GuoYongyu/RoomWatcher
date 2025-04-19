# coding: utf-8
from collections.abc import Sequence

import torch
from torch import nn
import numpy as np

from frame.frame import Frame
from logger import LOGGER

from detector.base import BaseDetector, DetectResult


class FasterRCNNDetector(BaseDetector):
    FRCNN_HELP_URL = "https://github.com/bubbliiiing/faster-rcnn-pytorch"
    FRCNN_NMS_IOU_THRESHOLD = 0.3


    def __init__(
            self,
            model_path: str,
            device: str | int = 0,
            # classes_path: str = None,
            # config_path: str = None,
            anchor_sizes: list = [8, 16, 32]
        ):
        super().__init__(
            det_method="FasterRCNN", device=device,
            model_path=model_path,
            # config_path=config_path, classes_path=classes_path
        )

        LOGGER.info(f"Use Faster RCNN PyTorch version to detect, details in: {self.FRCNN_HELP_URL}")

        from faster_rcnn.utils import DecodeBox

        self.model_path = model_path
        # self.classes_path = classes_path
        self.anchor_sizes = anchor_sizes
        self.device = device
        self.cuda = device.lower() != "cpu"

        self.num_classes = len(self.D_LABELS_MAP)

        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std = self.std.cuda()
        self.bbox_util = DecodeBox(self.std, self.num_classes)

        self.net = None
        self.__generate()


    def __generate(self):
        from faster_rcnn.faster_rcnn import FasterRCNN

        # load model
        self.net = FasterRCNN(
            num_classes=self.num_classes, mode="predict",
            anchor_scales=self.anchor_sizes
        )
        self.net.load_state_dict(
            torch.load(self.model_path, map_location=self.device),
            strict=False
        )
        self.net = self.net.eval()

        LOGGER.info(f"Success load model from: {self.model_path}")

        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()


    def __process(
            self,
            frame: Frame | np.ndarray = None,
            block_top: int = 0,
            block_left: int = 0
        ) -> dict:
        from faster_rcnn.utils import get_new_image_size, resize_image

        color_image = frame.frame_color if type(frame) == Frame else frame
        image_shape = color_image.shape[:2]
        # resize the image and get shape
        input_shape = get_new_image_size(image_shape[0], image_shape[1])

        image_data = resize_image(color_image, input_shape)
        # add batch_size dimension
        image_data  = np.expand_dims(np.transpose(np.array(image_data, dtype='float32') / 255.0, (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # roi_cls_locs  建议框的调整参数
            # roi_scores    建议框的种类得分
            # rois          建议框的坐标
            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            # 利用 classifier 的预测结果对建议框进行解码，获得预测框
            results = self.bbox_util.forward(
                roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                nms_iou=self.FRCNN_NMS_IOU_THRESHOLD, 
                confidence=self.D_CONFIDENCE_THRESHOLD
            )

            if len(results[0]) <= 0:
                return dict()
            
            top_label = np.array(results[0][:, 5], dtype = 'int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]    
        
        results_dict = dict((val, list()) for val in self.D_LABELS_MAP.keys())
        for i, c in enumerate(top_label):
            pred_cls = int(c)
            conf = top_conf[i]
            top, left, bottom, right = top_boxes[i]
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(color_image.shape[0], np.floor(bottom).astype('int32'))
            right = min(color_image.shape[1], np.floor(right).astype('int32'))

            if pred_cls not in self.D_LABELS_MAP:
                continue

            results_dict[pred_cls].append(DetectResult(
                label=self.D_LABELS_MAP[pred_cls], confidence=conf, 
                x1=int(left) + block_left, y1=int(top) + block_top, 
                x2=int(right) + block_left, y2=int(bottom) + block_top
            ))
        
        if len(results_dict) == 0:
            LOGGER.info("No Pre-defined Objects Found!")
        else:
            for cls in results_dict:
                LOGGER.warning(f"Found {len(results_dict[cls])} {self.D_LABELS_MAP[cls]}(s) in Current Frame/Image.")
                for info in results_dict[cls]:
                    LOGGER.warning(f"Position and Confidence: {info}")
        
        return results_dict


    def find_objects(
            self, 
            frame: Frame, 
            block_id: int | Sequence = None,
            specified_class: str | int = 0
        ) -> tuple[bool, list]:
        if block_id is None:
            results_dict = self.__process(frame)
        else:
            if type(block_id) == int:
                results_dict = self.__process(
                    frame.frame_blocks[block_id],
                    frame.frame_blocks[block_id].top,
                    frame.frame_blocks[block_id].left
                )
            else:
                results_dict = dict()
                for id in block_id:
                    res_dict = self.__process(
                        frame.frame_blocks[id],
                        frame.frame_blocks[id].top,
                        frame.frame_blocks[id].left
                    )
                    for label, vals in res_dict.items():
                        results_dict[label] += vals

        if type(specified_class) == str:
            for key, val in self.D_LABELS_MAP.items():
                if val == specified_class:
                    cls = key
        else:
            cls = specified_class

        if len(results_dict.get(cls, [])) > 0:
            return True, results_dict[cls]
        else:
            return False, []
        