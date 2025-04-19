# coding: utf-8
from collections.abc import Sequence

import torch
import numpy as np

from frame.frame import Frame
from logger import LOGGER

from detector.base import BaseDetector, DetectResult

# used in YOLOv5
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, check_img_size
# if use directory names with yolov5_models, yolov5_utils to train, then import sentences are below
# from yolov5_models.experimental import attempt_load
# from yolov5_utils.augmentations import letterbox
# from yolov5_utils.general import non_max_suppression, scale_coords, check_img_size


class Yolov5Detector(BaseDetector):
    def __init__(
            self,
            model_path: str = None,
            # classes_path: str = None,
            # config_path: str = None,
            device: str | int = 0
        ):
        super().__init__(
            det_method="YOLOv5-ST", device=device, 
            model_path=model_path,
            # config_path=config_path, classes_path=classes_path
        )

        self.model = attempt_load(self.model_path, map_location=self.device)


    def __process(
            self, 
            frame: Frame | np.ndarray = None,
            block_top: int = 0,
            block_left: int = 0
        ) -> dict:
        assert frame is not None, "When use detection, frame must be specified"

        results_dict = dict((val, list()) for val in self.D_LABELS_MAP.keys())

        if type(frame) == Frame:
            image_origin = frame.frame_color
        else:
            image_origin = frame
        
        stride = int(self.model.stride.max())
        image_size = check_img_size(640, s=stride)
        image = letterbox(image_origin, image_size, stride=stride)[0]
        image = image_origin.transpose(2, 0, 1)
        image = torch.from_numpy(np.ascontiguousarray(image) / 255.0)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        x_size, y_size = image.shape[2], image.shape[3]
        mulpicplus = 3
        x_smalloccur = int(x_size / mulpicplus * 1.2)
        y_smalloccur = int(y_size / mulpicplus * 1.2)
        for i in range(mulpicplus):
            x_start = int(i * (x_size / mulpicplus))
            for j in range(mulpicplus):
                y_start = int(j * (y_size / mulpicplus))
                x_real = min(x_start + x_smalloccur, x_size)
                y_real = min(y_start + y_smalloccur, y_size)
                if (x_real - x_start) % 64 != 0:
                    x_real = x_real - (x_real - x_start) % 64
                if (y_real - y_start) % 64 != 0:
                    y_real = y_real - (y_real - y_start) % 64
                dicsrc = image[:, :, x_start: x_real, y_start: y_real]

                temp_results = self.model(dicsrc.float(), augment=False, visualize=False)[0]
                temp_results[..., 0] = temp_results[..., 0] + y_start
                temp_results[..., 1] = temp_results[..., 1] + x_start

                if i == 0 and j == 0:
                    results = temp_results
                else:
                    results = torch.cat((results, temp_results), dim=1)

        results = non_max_suppression(
            results, 
            conf_thres=self.D_CONFIDENCE_THRESHOLD, 
            iou_thres=self.D_NMS_IOU_THRESHOLD
        )

        for result in results:
            if len(result):
                result[:, :4] = scale_coords(image.shape[2:], result[:, :4], image_origin.shape).round()

                for *xyxy, conf, cls in reversed(result):
                    cls = int(cls)
                    results_dict[cls].append(DetectResult(
                        label=self.D_LABELS_MAP[cls], confidence=conf, 
                        x1=int(xyxy[0]) + block_left, y1=int(xyxy[1]) + block_top, 
                        x2=int(xyxy[2]) + block_left, y2=int(xyxy[3]) + block_top
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
        