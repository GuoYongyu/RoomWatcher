# coding: utf-8
from collections.abc import Sequence

import cv2
import numpy as np

from frame.frame import Frame
from logger import LOGGER

from detector.base import DetectResult
        

class HOGPersonDetector(object):
    def __init__(
            self,
            window_stride: tuple[int, int] = (16, 16),
        ):
        self.window_stride = window_stride

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    
    def __process(
            self, 
            frame: Frame,
            block_top: int = 0,
            block_left: int = 0
        ) -> list:
        color_image = frame.frame_color if type(frame) == Frame else frame
        rects, weights = self.hog.detectMultiScale(
            color_image, winStride=self.window_stride
        )
        
        results_dict = {0: list()}
        for x, y, w, h, weight in zip(rects, weights):
            results_dict[0].append(DetectResult(
                label="person", confidence=weight,
                x1=int(x) + block_left, y1=int(y) + block_top, 
                x2=int(x + w) + block_left, y2=int(y + h) + block_top
            ))

        if len(results_dict[0]) > 0:
            LOGGER.warning(f"Found {len(results_dict[0])} person(s)!")
        else:
            LOGGER.warning("No Person Found!")
            
        return results_dict
    

    def find_persons(
            self, 
            frame: Frame | np.ndarray, 
            block_id: int | Sequence = None
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

        if len(results_dict.get(0, [])) > 0:
            return True, results_dict[0]
        else:
            return False, []
        

class GaussainMixture(object):
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    
    def __process(
            self, 
            frame: Frame | np.ndarray = None,
            block_top: int = 0,
            block_left: int = 0
        ) -> list:
        color_image = frame.frame_color if type(frame) == Frame else frame

        # remove noise
        fgmask = self.fgbg.apply(color_image)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results_dict = {0: list()}
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            results_dict[0].append(DetectResult(
                label="person", confidence=area / 5000.,
                x1=int(x) + block_left, y1=int(y) + block_top, 
                x2=int(x + w) + block_left, y2=int(y + h) + block_top
            ))

        if len(results_dict[0]) > 0:
            LOGGER.warning(f"Found {len(results_dict[0])} person(s)!")
        else:
            LOGGER.warning("No Person Found!")
            
        return results_dict
    

    def find_persons(
            self, 
            frame: Frame, 
            block_id: int | Sequence = None
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

        if len(results_dict.get(0, [])) > 0:
            return True, results_dict[0]
        else:
            return False, []
        

    def test(self, image: np.ndarray):
        return self.__process(image)
        