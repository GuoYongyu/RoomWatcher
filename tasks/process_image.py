# coding: utf-8
import os
import threading
import time
from typing import TypeAlias

import cv2
import numpy as np
from paddleocr import PaddleOCR

from frame.frame import Frame
from frame.utils import OCR_DETECTOR
from compare.compare import Comparer
from compare.result import CompareResult
from compare.utils import check_view_changed, revise_blocks_with_exceptions
from tasks.data_queue import IMAGE_FILES_QUEUE_MAP
from logger import LOGGER
from argparser import ArgParser, ArgSetter


_CompanyName: TypeAlias = str
_BlockID: TypeAlias = int
_SimilarityScore: TypeAlias = float


class ProcessImage(threading.Thread):
    def __init__(
            self,
            company_list: list[_CompanyName],
            yaml_path: str = "./config/cockroach-detection.yaml"
        ):
        super().__init__()

        self.name = "ProcessImage"

        self.company_list = company_list

        self.arguments = ArgParser(yaml_path)
        ArgSetter.set_args(self.arguments)

        self.comparer = Comparer(cmp_type=self.arguments["compare-type"])

        if Frame.F_DETECT_TEXT_IN_BLOCK:
            OCR_DETECTOR = PaddleOCR(use_angle_cls=True, use_gpu=self.arguments["gpu-device"].lower() == "cpu")

        if not os.path.exists("output"):
            os.makedirs("output")
            LOGGER.warning("Output directory 'output' created.")


    def join(self):
        return super().join()
    

    def __blocks_not_similar(
            self, 
            src_frames: dict[str, Frame], 
            tgt_frames: dict[str, Frame]
        ) -> tuple[dict[str, CompareResult], dict[str, _BlockID], dict[str, _SimilarityScore]]:
        sim_ret, ids_ret, sco_ret = dict(), dict(), dict()
        for device, tgt_frame in tgt_frames.items():
                src_frame = src_frames.get(device, None)
                if src_frame is None:
                    sim_ret[device], ids_ret[device], sco_ret[device] = list()
                else:
                    sim_infos, _, _ = self.comparer.get_similarity(src_frame, tgt_frame)
                    sim_ret[device] = sim_infos
                    ids_ret[device] = [res.block_id for res in sim_infos if not res.over_thresh]
                    sco_ret[device] = [res.similarity for res in sim_infos if not res.over_thresh]

        return sim_ret, ids_ret, sco_ret
    

    def run(self):
        first_frames = dict((c, dict()) for c in self.company_list)
        second_frames = dict((c, dict()) for c in self.company_list)
        current_frames = dict((c, dict()) for c in self.company_list)
        is_first_frame = dict((c, True) for c in self.company_list)
        is_second_frame = dict((c, False) for c in self.company_list)

        while True:
            for company in self.company_list:
                images_queue = IMAGE_FILES_QUEUE_MAP.get(company, None)
                if images_queue is None:
                    LOGGER.warning(f'{company} has no image obtained yet at ' + \
                                   f'{time.strftime("%Y%m%d %H:%M:%S", time.localtime())}')
                    continue

                if is_first_frame[company]:
                    image_list = images_queue.get()
                    first_frames[company] = dict(
                        (img[0], Frame(image=img[1], id_prefix=img[0], to_rgb=False))
                        for img in image_list
                    )
                    is_first_frame[company] = False
                    is_second_frame[company] = True
                    continue
                elif is_second_frame[company]:
                    image_list = images_queue.get()
                    first_frames[company] = dict(
                        (img[0], Frame(image=img[1], id_prefix=img[0], to_rgb=False))
                        for img in image_list
                    )
                    is_second_frame[company] = False
                    _, block_ids_first_second, _ = \
                        self.__blocks_not_similar(first_frames[company], second_frames[company])
                    continue
                else:
                    image_list = images_queue.get()
                    current_frames[company] = dict(
                        (img[0], Frame(image=img[1], id_prefix=img[0], to_rgb=False))
                        for img in image_list
                    )
                    siminfos_second_curr, block_ids_second_curr, _ = \
                        self.__blocks_not_similar(second_frames[company], current_frames[company])
                    
                    for device, curr_frame in current_frames[company].items():
                        second_frame = second_frames[company][device]
                        if curr_frame.may_have_exception:
                            if Comparer.C_CHECK_VIEW_CHANGED and check_view_changed(siminfos_second_curr[device]):
                                image = curr_frame.draw_results(text="View Changed", loc="left-bottom")
                                LOGGER.warning(f"Detected view changed in {company}-{device}.")
                                image = np.concatenate([second_frames[company][device].frame_color, image], axis=1)
                                cv2.imwrite(os.path.join("output", f"viewchange_{company}_{device}.jpg"), image)
                            elif second_frame.may_have_exception:
                                block_ids = revise_blocks_with_exceptions(
                                    block_ids_first_second[device], block_ids_second_curr[device]
                                )
                                image = curr_frame.draw_results(block_ids=block_ids)
                                LOGGER.warning(f"Perhaps detected exception in {company}-{device}.")
                                cv2.imwrite(os.path.join("output", f"exception_{company}_{device}.jpg"), image)
                            else:
                                image = curr_frame.draw_results(block_ids=block_ids_second_curr[device])
                                LOGGER.info(f"Perhaps detected exception in {company}-{device}.")
                                cv2.imwrite(os.path.join("output", f"exception_{company}_{device}.jpg"), image)

                images_queue.task_done()
            
            first_frames[company] = second_frames[company]
            second_frames[company] = current_frames[company]
            block_ids_first_second = block_ids_second_curr
