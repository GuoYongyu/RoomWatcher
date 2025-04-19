# coding: utf-8
import time

import numpy as np

from frame.frame import Frame
from logger import LOGGER

from detector.base import BaseDetector
from detector.yolov5 import Yolov5Detector
from detector.yolov8 import Yolov8Detector
from detector.yolov10 import Yolov10Detector
from detector.fasterrcnn import FasterRCNNDetector


class Detector(object):
    DT_YOLOV5_SMALL_TARGET    = "yolov5-st"
    DT_YOLOV8_SMALL_TARGET    = "yolov8-st"
    DT_YOLOV8_NORMAL_EDITION  = "yolov8"
    DT_YOLOV10_NORMAL_EDITION = "yolov10"
    DT_FASTER_R_CNN           = "fasterrcnn"

    DT_IMAGE_PROCESS_DEVICE   = "0"


    def __init__(
            self, 
            det_method: str = "YOLOv8", 
            model_path: str = None,
            # config_path: str = None,
            # device: str | int = 0,
        ):
        detector_classes = {
            Detector.DT_YOLOV10_NORMAL_EDITION: Yolov10Detector,
            Detector.DT_YOLOV8_NORMAL_EDITION:  Yolov8Detector,
            Detector.DT_YOLOV8_SMALL_TARGET:    Yolov8Detector,
            Detector.DT_YOLOV5_SMALL_TARGET:    Yolov5Detector,
            Detector.DT_FASTER_R_CNN:           FasterRCNNDetector,
        }
        if det_method.lower() not in detector_classes:
            LOGGER.error(f"Unknown detector method '{det_method}'")
            raise NotImplementedError(f"Unknown detector method '{det_method}'")

        self.__detector = detector_classes[det_method](
            model_path=model_path, 
            # config_path=config_path,
            device=self.DT_IMAGE_PROCESS_DEVICE
        )
        self.__warm_up(BaseDetector.D_WARM_UP_TIMES)

    
    def __warm_up(self, times: int = 1):
        LOGGER.info(f"Warming up the detector {times} times ...")
        LOGGER.disabled = True

        start_time = time.monotonic_ns()
        try:
            for _ in range(times):
                self.__detector.find_objects(Frame(np.zeros([720, 1280, 3])))
        except Exception as e:
            LOGGER.error(f"Failed to warm up the detector: {e}")
        finally:
            LOGGER.disabled = False
        end_time = time.monotonic_ns()
        LOGGER.info(f"Warming up took {(end_time - start_time) / 1e6} ms.")


    def find_objects(self, frame: Frame) -> tuple[bool, list]:
        start_time = time.monotonic_ns()
        results = self.__detector.find_objects(frame=frame)
        end_time = time.monotonic_ns()
        LOGGER.info(f"Detection took {(end_time - start_time) / 1e6} ms.")

        return results
