# coding: utf-8
from collections.abc import Sequence

import numpy as np
from ultralytics import YOLO

from frame.frame import Frame
from logger import LOGGER

from detector.base import BaseDetector, DetectResult


class Yolov8Detector(BaseDetector):
    def __init__(
            self,
            model_path: str = None,
            # classes_path: str = None,
            # config_path: str = None,
            device: str | int = 0
        ):
        super().__init__(
            det_method="YOLOv8", device=device, 
            model_path=model_path,
            # config_path=config_path, classes_path=classes_path
        )

        self.model = YOLO(self.model_path, verbose=False)


    def __process(
            self, 
            frame: Frame | np.ndarray = None,
            block_top: int = 0,
            block_left: int = 0
        ) -> dict:
        assert frame is not None, "When use detection, frame must be specified"

        results_dict = dict((val, list()) for val in self.D_LABELS_MAP.keys())

        if type(frame) == Frame:
            results = self.model(frame.frame_color, save=False, device=self.device)
        else:
            results = self.model(frame, save=False, device=self.device)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                label = int(box.cls)
                xyxy = box.xyxy.cpu().tolist()[0]

                if self.D_USE_CONFIDENCE_THRESHOLD and conf <= self.D_CONFIDENCE_THRESHOLD:
                    continue

                results_dict[label].append(DetectResult(
                    label=self.D_LABELS_MAP[label], confidence=conf, 
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
        