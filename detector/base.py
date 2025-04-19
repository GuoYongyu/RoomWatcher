# coding: utf-8
import os

import torch

from logger import LOGGER


class DetectResult(object):
    def __init__(
            self,
            x1: int,
            x2: int,
            y1: int,
            y2: int,
            confidence: float,
            label: str,
        ):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.width = abs(self.x2 - self.x1)
        self.height = abs(self.y2 - self.y1)
        self.confidence = confidence
        self.label = label


    def __str__(self) -> str:
        return f"label = {self.label}, " + \
               f"confidence = {round(self.confidence, 3)}, " + \
               f"x1 = {self.x1}, y1 = {self.x2}, " + \
               f"x2 = {self.y1}, y2 = {self.y2}, " + \
               f"width = {self.width}, height = {self.height}"


class BaseDetector(object):
    D_CONFIDENCE_THRESHOLD     = 0.2
    D_NMS_IOU_THRESHOLD        = 0.45
    D_USE_CONFIDENCE_THRESHOLD = True

    D_LABELS_MAP = {
        0: "mouse",
        1: "person"
    }

    D_WARM_UP_TIMES = 1


    def __init__(
            self, 
            det_method: str = "YOLOv8",
            device: str | int = 0,
            model_path: str = None,
            # classes_path: str = None,
            # config_path: str = None,
        ):
        self.det_method = det_method
        LOGGER.info(f"Use {self.det_method} as detector")

        assert model_path is not None, "model path must be specified"

        if device != "cpu" and not torch.cuda.is_available():
            LOGGER.warning("CUDA is not available, use CPU instead.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model_path = model_path
        # self.config_path = config_path
        LOGGER.info(f"Path to detector model: {os.path.abspath(model_path)}")
