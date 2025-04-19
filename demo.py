# coding: utf-8
import os

import cv2
import numpy as np
from paddleocr import PaddleOCR

from frame.frame import Frame
from frame.utils import OCR_DETECTOR
from compare.compare import Comparer
from compare.utils import check_view_changed, revise_blocks_with_exceptions
from argparser import ArgSetter, ArgParser
from detect import Detector
from logger import LOGGER


class DemoBase(object):
    def __init__(
            self, 
            demo_way: str = "compare-based",
            yaml_path: str = None
        ):
        self.demo_way = demo_way.lower()

        self.arguments = ArgParser().arguments if yaml_path is None else ArgParser(yaml_path).arguments
        ArgSetter.set_args(self.arguments)

        if isinstance(self.arguments["compare-type"], list):
            self.arguments["compare-type"] = self.arguments["compare-type"][0]
            LOGGER.warning(f"Found {len(self.arguments)} compare types, only using the first one: {self.arguments['compare-type']} in DEMO.")
        self.comparer = Comparer(cmp_type=self.arguments["compare-type"])

        if Frame.F_DETECT_TEXT_IN_BLOCK:
            OCR_DETECTOR = PaddleOCR(use_angle_cls=True, use_gpu=self.arguments["gpu-device"].lower() == "cpu")

        # read video from file
        assert self.arguments["video-path"] is not None, "No video path provided."
        self.capture = cv2.VideoCapture(self.arguments["video-path"])
        if not self.capture.isOpened():
            LOGGER.error(f'Cannot open video from {self.arguments["video-path"]}.')
            raise ValueError("Cannot open video.")
        LOGGER.info(f'Video capture from {self.arguments["video-path"]} is OK.')

        # total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # logger.info(f"Total frames of the video: {total_frames}")
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        LOGGER.info(f'The FPS of video is {fps}.')
        self.time_per_frame = 1 / fps


    def run(self):
        LOGGER.info(f"Way to demo is {self.demo_way}")


class MixtureDemo(DemoBase):
    def __init__(self, yaml_path: str = None):
        super().__init__(demo_way="mixture", yaml_path=yaml_path)


    def run(self):
        LOGGER.info(f'Running demo with {self.demo_way}...')
        
        assert self.arguments["use-detector"], "Use detector in Mixture Demo but related arguments not provided."
        det_method = self.arguments["detector-method"]
        detector = Detector(
            det_method=det_method,
            model_path=self.arguments[f"{det_method.lower()}-model"], 
            # device=self.arguments["gpu-device"]
        )

        is_first_frame, frame_count, discard_count = True, 0, 0
        while True:
            ret, frame_ori = self.capture.read() 
            if not ret:
                LOGGER.warning(f'Finished reading frame from {self.arguments["video-path"]}.')
                break
            
            # discard first 30 frames
            discard_count += 1
            if discard_count <= 30:
                continue

            frame_count += 1

            if is_first_frame:
                last_frame = Frame(image=frame_ori, to_rgb=self.arguments["transfer-rgb"])
                is_first_frame = False
                frame_count = 1
                continue
            elif frame_count * self.time_per_frame >= self.arguments["video-sample-gap-sec"]:
                curr_frame = Frame(image=frame_ori, to_rgb=self.arguments["transfer-rgb"])
                _ = self.comparer.get_similarity(last_frame, curr_frame)
                if curr_frame.may_have_exception:
                    if self.arguments["use-detector"]:
                        has_enermy, det_res = detector.find_objects(curr_frame)
                        if has_enermy:
                            LOGGER.warning(f'Perhaps found {len(det_res)} exception objects in Frame-{curr_frame.frame_id}.')
                    else:
                        LOGGER.info(f'It seems no exception object in Frame-{curr_frame.frame_id}.')
                else:
                    LOGGER.info(f'It seems no exception object in Frame-{curr_frame.frame_id}.')

                last_frame = curr_frame
                frame_count = 0

        self.capture.release()


class DirectDetectionDemo(DemoBase):
    def __init__(self, yaml_path: str = None):
        super().__init__(demo_way="direct-detection", yaml_path=yaml_path)


    def run(self):
        LOGGER.info(f'Running demo with {self.demo_way}...')

        assert self.arguments["use-detector"], "Use detector in Mixture Demo but related arguments not provided."
        det_method = self.arguments["detector-method"]
        detector = Detector(
            det_method=det_method,
            model_path=self.arguments[f"{det_method.lower()}-model"], 
            # device=self.arguments["gpu-device"]
        )

        discard_count, frame_count = 0, 0
        while True:
            ret, frame_ori = self.capture.read()
            if not ret:
                LOGGER.warning(f'Finished reading frame from {self.arguments["video-path"]}.')
                break

            # discard first 30 frames
            discard_count += 1
            if discard_count <= 30:
                continue
                
            frame_count += 1

            cur_frame = Frame(image=frame_ori, blocking=False, to_rgb=self.arguments["transfer-rgb"])
            has_enermy, det_res = detector.find_objects(cur_frame)
            if has_enermy:
                image = cur_frame.draw_results(det_boxes=det_res)
                LOGGER.warning(f'Perhaps found {len(det_res)} exception objects in Frame-{cur_frame.frame_id}.')
            else:
                image = cur_frame.frame_color
                LOGGER.info(f'It seems no exception object in Frame-{cur_frame.frame_id}.')

            cv2.imshow('Mouse Detector Demo', image)
            cv2.waitKey(1)

        self.capture.release()


class CompareBasedDemo(DemoBase):
    def __init__(self, yaml_path: str = None):
        super().__init__(demo_way="compare-based", yaml_path=yaml_path)


    def run(self):
        LOGGER.info(f'Running demo with {self.demo_way}...')

        is_first_frame, frame_count, discard_count = True, 0, 0
        while True:
            ret, frame_ori = self.capture.read()
            if not ret:
                LOGGER.warning(f'Finished reading frame from {self.arguments["video-path"]}.')
                break

            # discard first 100 frames
            discard_count += 1
            if discard_count <= 100:
                continue
                
            frame_count += 1

            if is_first_frame:
                # initialize first frame
                last_frame = Frame(image=frame_ori, to_rgb=self.arguments["transfer-rgb"])
                is_first_frame = False
                frame_count = 1
                continue
            elif frame_count * self.time_per_frame >= self.arguments["video-sample-gap-sec"]:
                curr_frame = Frame(image=frame_ori, to_rgb=self.arguments["transfer-rgb"])
                sim_reses, _ = self.comparer.get_similarity(last_frame, curr_frame)
                if curr_frame.may_have_exception:
                    if Comparer.C_CHECK_VIEW_CHANGED and check_view_changed(sim_reses, curr_frame):
                        image = curr_frame.draw_results(text="View Changed")
                        LOGGER.warning(f'Detected view change in Frame-{curr_frame.frame_id}.')
                        written_image = np.concatenate([last_frame.frame_color, curr_frame.frame_color], axis=1)
                        cv2.imwrite(os.path.join("output", f"viewchange_{curr_frame.frame_id}.jpg"), written_image)
                        LOGGER.info(f"View changed frame has been saved as output/viewchange_{curr_frame.frame_id}.jpg")
                    elif type(sim_reses) == list:
                        block_ids = [res.block_id for res in sim_reses if not res.over_thresh]
                        image = curr_frame.draw_results(block_ids=block_ids)
                        LOGGER.warning(f'Perhaps found exception object in Frame-{curr_frame.frame_id}.')
                        cv2.imwrite(os.path.join("output", f"excp_{curr_frame.frame_id}.jpg"), image)
                        LOGGER.info(f"Frame has been saved as output/excp_{curr_frame.frame_id}.jpg")
                    else:
                        image = curr_frame.draw_results(text="Found Exception")
                        LOGGER.warning(f'Perhaps found exception object in Frame-{curr_frame.frame_id}.')
                        cv2.imwrite(os.path.join("output", f"excp_{curr_frame.frame_id}.jpg"), image)
                        LOGGER.info(f"Frame has been saved as output/excp_{curr_frame.frame_id}.jpg")
                else:
                    image = curr_frame.frame_color
                    LOGGER.info(f'It seems no exception object in Frame-{curr_frame.frame_id}.')

                last_frame = curr_frame
                frame_count = 0

                cv2.imshow('Compare based Demo', image)
                cv2.waitKey(1)

        self.capture.release()


class CompareImprovedDemo(DemoBase):
    def __init__(self, yaml_path: str = None):
        super().__init__(demo_way="compare-improved", yaml_path=yaml_path)


    def run(self):
        LOGGER.info(f'Running demo with {self.demo_way}...')

        is_first_frame, is_second_frame = True, False
        frame_count, discard_count = 0, 0
        
        while True:
            ret, frame_ori = self.capture.read()
            if not ret:
                LOGGER.warning(f'Finished reading frame from {self.arguments["video-path"]}.')
                break

            # discard first 100 frames
            discard_count += 1
            if discard_count <= 100:
                continue
            
            # sample every n frames
            frame_count += 1
            if frame_count * self.time_per_frame < self.arguments["video-sample-gap-sec"]:
                continue
            frame_count = 0

            if is_first_frame:
                # initialize first frame
                first_frame = Frame(image=frame_ori, to_rgb=self.arguments["transfer-rgb"])
                is_first_frame = False
                is_second_frame = True
                continue
            elif is_second_frame:
                second_frame = Frame(image=frame_ori, to_rgb=self.arguments["transfer-rgb"])
                is_second_frame = False
                sim_first_second, _ = self.comparer.get_similarity(first_frame, second_frame)
                block_ids_first_second = [res.block_id for res in sim_first_second if not res.over_thresh]
                continue
            else:
                curr_frame = Frame(image=frame_ori, to_rgb=self.arguments["transfer-rgb"])
                sim_second_curr, mask = self.comparer.get_similarity(second_frame, curr_frame)
                block_ids_second_curr: list[int] = [res.block_id for res in sim_second_curr if not res.over_thresh]
                scores_second_curr: list[float] = [res.similarity for res in sim_second_curr if not res.over_thresh]
                if curr_frame.may_have_exception:
                    if Comparer.C_CHECK_VIEW_CHANGED and check_view_changed(sim_second_curr, 0.8, mask):
                        image = curr_frame.draw_results(text="View Changed", text_loc="left-bottom")
                        LOGGER.warning(f'Detected view change in Frame-{curr_frame.frame_id}.')
                        written_image = np.concatenate([second_frame.frame_color, curr_frame.frame_color], axis=1)
                        cv2.imwrite(os.path.join("output", f"viewchange_{curr_frame.frame_id}.jpg"), written_image)
                    elif second_frame.may_have_exception:
                        block_ids = revise_blocks_with_exceptions(block_ids_first_second, block_ids_second_curr)
                        scores = [sim_second_curr[id].similarity for id in block_ids]
                        image = curr_frame.draw_results(block_ids=block_ids, block_scores=scores, mask=mask)
                        LOGGER.warning(f'Perhaps found exception object in Frame-{curr_frame.frame_id}.')
                        cv2.imwrite(os.path.join("output", f"excp_{curr_frame.frame_id}.jpg"), image)
                        LOGGER.info(f"Frame has been saved to output/excp_{curr_frame.frame_id}.jpg")
                    else:
                        image = curr_frame.draw_results(block_ids=block_ids_second_curr, block_scores=scores_second_curr, mask=mask)
                        LOGGER.warning(f'Perhaps found exception object in Frame-{curr_frame.frame_id}.')
                        cv2.imwrite(os.path.join("output", f"excp_{curr_frame.frame_id}.jpg"), image)
                        LOGGER.info(f"Frame has been saved to output/excp_{curr_frame.frame_id}.jpg")
                else:
                    image = curr_frame.frame_color
                    LOGGER.info(f'It seems no exception object in Frame-{curr_frame.frame_id}.')

            first_frame = second_frame
            second_frame = curr_frame
            block_ids_first_second = block_ids_second_curr

            cv2.imshow('Compare Improved Demo', image)
            cv2.waitKey(1)
        
        self.capture.release()


class Demoer(object):
    DM_MIXTURE          = "mixture"
    DM_DIRECT_DETECTION = "direct-detection"
    DM_COMPARE_BASED    = "compare-based"
    DM_COMPARE_IMPROVED = "compare-improved"


    def __init__(self, demo_way: str, yaml_path: str = None):
        demo_classes = {
            Demoer.DM_MIXTURE:           MixtureDemo,
            Demoer.DM_DIRECT_DETECTION:  DirectDetectionDemo,
            Demoer.DM_COMPARE_BASED:     CompareBasedDemo,
            Demoer.DM_COMPARE_IMPROVED:  CompareImprovedDemo,
        }
        if demo_way not in demo_classes:
            raise NotImplementedError(f"Wrong input for demo way: {demo_way}.")
        
        if demo_way != Demoer.DM_COMPARE_BASED:
            ## text detection only for compare-based
            Frame.F_DETECT_TEXT_IN_BLOCK = False

        self.__demoer = demo_classes[demo_way](yaml_path)
        LOGGER.info(f"Use {demo_way} to demo.")


    def run(self):
        self.__demoer.run()
