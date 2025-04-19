# coding: utf-8
import os
import time
import time
from typing import TypeAlias
from collections.abc import Iterable

import cv2
import numpy as np

from logger import LOGGER
from frame.colors import RECT_COLORS
from frame.utils import OCR_DETECTOR, find_text_boxes, block_with_text, merge_intersecting_blocks


FRAME_COUNT = {
    "default": 0,
}


_ImagePath: TypeAlias = str


class FrameBlock(object):
    FB_CAL_KPS_DESC = False


    def __init__(
            self, image: np.ndarray,
            frame_id: int, block_id: int,
            top: int = None, left: int = None,
            width: int = None, height: int = None,
        ):
        self._frame_id = frame_id
        self._block_id = block_id

        self._top    = top
        self._left   = left
        self._width  = width
        self._height = height

        self.keypoints   = None
        self.descriptors = None

        self.has_text = False
        
        if FrameBlock.FB_CAL_KPS_DESC:
            self.__cal_keypoints_and_descriptors(image[top: top + height, left: left + width])

    
    @property
    def frame_id(self) -> int:
        return self._frame_id


    @property
    def block_id(self) -> int:
        return self._block_id


    @property
    def top(self) -> int:
        return self._top


    @property
    def left(self) -> int:
        return self._left


    @property
    def width(self) -> int:
        return self._width


    @property
    def height(self) -> int:
        return self._height


    def __cal_keypoints_and_descriptors(self, image: np.ndarray):
        cal_functions = {
            "ORB":  self.__cal_kp_des_with_orb,
            "FAST": self.__cal_kp_des_with_fast
        }
        if Frame.F_KEYPOINT_TYPE not in cal_functions:
            LOGGER.error(f"Unsupported keypoint type: {Frame.F_KEYPOINT_TYPE}.")
            raise Exception("Unsupported keypoint type.")
        
        cal_func = cal_functions[Frame.F_KEYPOINT_TYPE]
        cal_func(image)
        

    def __cal_kp_des_with_orb(self, image: np.ndarray):
        orb = cv2.ORB_create(nfeatures=Frame.F_KEYPOINT_NUM_PER_BLOCK)
        self.keypoints, self.descriptors = orb.detectAndCompute(image, None)

        if self.keypoints is None or self.descriptors is None:
            LOGGER.warning(f"Failed to get keypoints and descriptors in " + 
                           f"Block {self.block_id} (Frame {self.frame_id}).")
            self.keypoints   = None
            self.descriptors = None


    def __cal_kp_des_with_fast(self, image: np.ndarray):
        fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
        self.keypoints = fast.detect(image, None)

        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.keypoints, self.descriptors = brief.compute(image, self.keypoints)

        if self.keypoints is None or self.descriptors is None:
            LOGGER.warning(f"Failed to get keypoints and descriptors in " + 
                           f"Block {self.block_id} (Frame {self.frame_id}).")
            self.keypoints   = None
            self.descriptors = None


class Frame(object):
    F_STRIP_RATE_INV         = 18     # 图像块的重叠边长占图像总边长比例的倒数
    F_SPLIT_RATE_INV         = 9      # 图像块边长占图像总边长的比例的倒数

    F_MERGE_BLOCKS           = True   # 当多个块重叠时是否合并，用于绘制结果使用

    F_KEYPOINT_TYPE          = "ORB"
    F_KEYPOINT_NUM_PER_BLOCK = 200

    F_DETECT_TEXT_IN_BLOCK   = False  # 是否检测图像块中的文本
    # F_OCR_WITH_GPU           = False  # 是否使用 GPU 进行 OCR

    F_RESULTS_PRESENT        = 1      # 结果呈现，1 仅可视化，2 仅保存，3 可视化并保存
    F_DET_CMP_METHOD         = "PIXEL-DIFF"


    def __init__(
            self, image: np.ndarray | _ImagePath, 
            id_prefix: str = "default", 
            blocking: bool = True, 
            to_rgb: bool = True
        ):
        """
            image (np.ndarray): 输入的图像数据，类型为 numpy 数组。
            id_prefix (str, optional): 帧的 ID 前缀，默认为 default。
            blocking (bool, optional): 是否将帧分割为块，默认为 True。
            to_rgb (bool, optional): 是否将图像从 BGR 转换为 RGB 格式，默认为 True。
        """
        if isinstance(image, _ImagePath):
            image = cv2.imread(image)

        if len(image.shape) == 2:
            # 灰度图像
            self.frame_color = self.frame_gray = image
        else:
            self.frame_color = image
            if to_rgb:
                self.frame_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.frame_gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.frame_height, self.frame_width = self.frame_color.shape[:2]

        self.frame_blocks: list[FrameBlock] = list()

        global FRAME_COUNT
        FRAME_COUNT[id_prefix] += 1
        self.frame_id: str = f"{id_prefix}-{FRAME_COUNT[id_prefix]}"

        self.timestamp: float = time.mktime(time.localtime())

        self.may_have_exception: bool = False

        self.text_boxes: list[list[int]] = None
        if Frame.F_DETECT_TEXT_IN_BLOCK:
            start_time = time.monotonic_ns()
            self.text_boxes = find_text_boxes(self.frame_color, OCR_DETECTOR)
            use_time = (time.monotonic_ns() - start_time) / 1e6
            LOGGER.info(f"Frame {self.frame_id} has detected {len(self.text_boxes)} text boxes within {use_time}ms.")

        if blocking:
            self.__split_frame_to_blocks()
            LOGGER.info(f"Frame {self.frame_id} has created {len(self.frame_blocks)} frame blocks.")


    def __split_frame_to_blocks(self):
        assert Frame.F_STRIP_RATE_INV == 2 * Frame.F_SPLIT_RATE_INV, "strip-rate must be twice of split rate"

        top, left, block_id = 0, 0, 0
        top_step, left_step = \
            self.frame_height // Frame.F_STRIP_RATE_INV, \
            self.frame_width // Frame.F_STRIP_RATE_INV
        
        for i in range(Frame.F_STRIP_RATE_INV - 1):
            height = int(self.frame_height / Frame.F_SPLIT_RATE_INV)
            if i == Frame.F_STRIP_RATE_INV - 2:
                    height = self.frame_height - top

            for j in range(Frame.F_STRIP_RATE_INV - 1):
                width  = int(self.frame_width / Frame.F_SPLIT_RATE_INV)
                if j == Frame.F_STRIP_RATE_INV - 2:
                    width = self.frame_width - left

                self.frame_blocks.append(
                    FrameBlock(
                        self.frame_color, self.frame_id, 
                        block_id, top, left, width, height
                    )
                )
                if Frame.F_DETECT_TEXT_IN_BLOCK and block_with_text(self.text_boxes, [left, top, width, height]):
                    self.frame_blocks[-1].has_text = True

                left += left_step
                block_id += 1
            top += top_step
            left = 0


    def draw_results(
            self, 
            text: str = None,
            text_loc: str = "left-top",             # bind with text
            det_boxes: Iterable[dict] = None,
            block_ids: Iterable[int] = None,
            block_scores: Iterable[float] = None,   # bind with block_ids
            rectangles: Iterable[Iterable] = None,
            mask: np.ndarray = None,                # optional
        ) -> np.ndarray:
        if text is not None:
            image = self.__draw_exception_text(text, text_loc, mask)
        elif det_boxes is not None:
            image = self.__draw_det_boxes(det_boxes, mask)
        elif block_ids is not None:
            image = self.__draw_blocks(block_ids, block_scores, mask)
        elif rectangles is not None:
            image = self.__draw_rectangels(rectangles, mask)
        else:
            image = self.frame_color

        return image


    def __draw_blocks(self, block_ids: Iterable = None, scores: Iterable = None, mask: np.ndarray = None) -> np.ndarray:
        """drow frame blocks with specified ids"""
        block_ids = block_ids if block_ids is not None else list(range(len(self.frame_blocks)))
        assert len(block_ids) == len(scores)

        rectangles = list()
        for id in block_ids:
            block: FrameBlock = self.frame_blocks[id]
            rectangles.append((block.left, block.top, block.left + block.width, block.top + block.height))
        
        if Frame.F_MERGE_BLOCKS:
            merged_rectangles = merge_intersecting_blocks(rectangles)
        else:
            merged_rectangles = rectangles
        
        color_image = self.frame_color.copy()
        if mask is not None:
            assert len(mask.shape) == 2, "mask only have one channel"
            color_image[mask > 0] = (59, 233, 250)

        if scores is not None:
            for score, rect in zip(scores, rectangles):
                cv2.putText(color_image, f"{score:.2f}", (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.rectangle(color_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 1)

        for i, rect in enumerate(merged_rectangles):
            cv2.rectangle(color_image, (rect[0], rect[1]), (rect[2], rect[3]), RECT_COLORS[i], 2)

        if Frame.F_RESULTS_PRESENT > 1:
            subdir = os.path.join("./results", self.frame_id.split("-")[0])
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            image_file = os.path.join(subdir, f"exception_{self.frame_id}_{self.timestamp}.jpg")
            cv2.imwrite(image_file, color_image)
            LOGGER.info(f"Frame {self.frame_id} has been saved to {image_file}.")
            
        return color_image
            

    def __draw_det_boxes(self, det_boxes: Iterable[dict], mask: np.ndarray = None) -> np.ndarray:
        """draw bounding boxes"""
        color_image = self.frame_color.copy()
        if mask is not None:
            color_image = cv2.bitwise_and(color_image, mask)

        for i, box in enumerate(det_boxes):
            conf = box["confidence"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), RECT_COLORS[i], 3)
            cv2.putText(color_image, f"{conf:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RECT_COLORS[i], 2)

        if Frame.F_RESULTS_PRESENT > 1:
            subdir = os.path.join("./results", self.frame_id.split("-")[0])
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            image_file = os.path.join(subdir, f"objects_{self.frame_id}_{self.timestamp}.jpg")
            cv2.imwrite(image_file, color_image)
            LOGGER.info(f"Frame {self.frame_id} has been saved to {image_file}.")

        return color_image
    

    def __draw_exception_text(
            self, text: str = "Found Exception", 
            loc: str = ["left-top", "left-bottom", "right-top", "right-bottom"],
            mask: np.ndarray = None
        ):
        color_image = self.frame_color.copy()
        if mask is not None:
            color_image = cv2.bitwise_and(color_image, mask)

        if loc == "left-top":
            org = (40, 40)
        elif loc == "left-bottom":
            org = (40, self.frame_height - 80)
        elif loc == "right-top":
            org = (self.frame_width - 10 * len(text), 40)
        elif loc == "right-bottom":
            org = (self.frame_width - 10 * len(text), self.frame_height - 80)
        else:
            LOGGER.error(f"Invalid location value '{loc}'.")
            raise ValueError("Invalid location value.")
        
        cv2.putText(color_image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        if Frame.F_RESULTS_PRESENT > 1:
            subdir = os.path.join("./results", self.frame_id.split("-")[0])
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            text = "_".join(text.lower().split(" "))
            image_file = os.path.join(subdir, f"{text}_{self.frame_id}_{self.timestamp}.jpg")
            cv2.imwrite(image_file, color_image)
            LOGGER.info(f"Frame {self.frame_id} has been saved to {image_file}.")

        return color_image
    

    def __draw_rectangels(self, rectangles: Iterable[Iterable], mask: np.ndarray):
        color_image = self.frame_color.copy()
        if mask is not None:
            color_image = cv2.bitwise_and(color_image, mask)

        for i, rect in enumerate(rectangles):
            cv2.rectangle(color_image, (rect[0], rect[1]), (rect[2], rect[3]), RECT_COLORS[i], 2)

        if Frame.F_RESULTS_PRESENT > 1:
            subdir = os.path.join("./results", self.frame_id.split("-")[0])
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            image_file = os.path.join(subdir, f"exception_{self.frame_id}_{self.timestamp}.jpg")
            cv2.imwrite(image_file, color_image)
            LOGGER.info(f"Frame {self.frame_id} has been saved to {image_file}.")

        return color_image


    def print_blocks(self):
        for i, block in enumerate(self.frame_blocks):
            print(f"Frame {self.frame_id} / Block {i + 1}: [{block.top}, {block.left}] - [{block.height}, {block.width}]")


    def draw_keypoints(self):
        image = self.frame_color.copy()
        keypoints = list()
        for block in self.frame_blocks:
            if block.keypoints is None: 
                continue

            t_keypoints = list()
            for kp in block.keypoints:
                kp_new = cv2.KeyPoint(
                    kp.pt[0] + block.left, kp.pt[1] + block.top, 
                    kp.size, kp.angle, kp.response, kp.octave, kp.class_id
                )
                t_keypoints.append(kp_new)
            keypoints += t_keypoints
        
        print(f"Frame {self.frame_id} has {len(keypoints)} keypoints.")
        cv2.drawKeypoints(image, keypoints, None)
        
        return image
