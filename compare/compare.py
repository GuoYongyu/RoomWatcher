# coding: utf-8
import time
from typing import Callable, TypeAlias

import cv2
import numpy as np
from skimage.metrics import structural_similarity as calc_structure_similarity

from frame.frame import Frame
from logger import LOGGER
from compare.utils import calc_hist_similarity, calc_template_similarity, calc_avg_hash_similarity
from compare.utils import revised_similarity_when_merging_blocks
from compare.utils import block_ids_has_rect
from compare.utils import calc_outlier_in_mask
from compare.task_functions import ThreadingBlocksCompareTask, BlocksCompareTask
from compare.result import CompareResult


_ImageMask: TypeAlias = np.ndarray


class Comparer(object):
    C_IMAGE_SIMILARITY_THRESHOLD   = 0.95
    C_UNMATCHED_BLOCKS_THRESHOLD   = 1
    C_MATCHED_KEYPOINTS_THRESHOLD  = 0.95

    C_CHECK_VIEW_CHANGED = True
    C_MULTI_PROCESSING   = 0

    C_COMPUTE_OUTLIERS   = "MEAN"

    # compare types
    CT_KEY_POINTS       = "KEY-POINTS"
    CT_PIXEL_DIFF       = "PIXEL-DIFF"
    CT_SSIM_BLOCKS      = "SSIM-BLOCKS"
    CT_SSIM_WHOLE       = "SSIM-WHOLE"
    CT_HIST_BLOCKS      = "HIST-BLOCKS"
    CT_HIST_WHOLE       = "HIST-WHOLE"
    CT_TEMPLATE_BLOCKS  = "TEMPLATE-BLOCKS"
    CT_TEMPLATE_WHOLE   = "TEMPLATE-WHOLE"
    CT_AVG_HASH_BLOCKS  = "AVG-HASH-BLOCKS"
    CT_AVG_HASH_WHOLE   = "AVG-HASH-WHOLE"
    

    def __init__(
            self, cmp_type: str = "KEY-POINTS"
        ):
        self.compare_type = cmp_type.upper()


    def get_similarity(self, src_frame: Frame, tgt_frame: Frame) -> tuple[list[CompareResult], _ImageMask | None]:
        compare_functions = {
            Comparer.CT_KEY_POINTS:       (self.__compare_with_keypoints,  None),
            Comparer.CT_PIXEL_DIFF:       (self.__compare_with_pixel_diff, None),
            Comparer.CT_SSIM_BLOCKS:      (self.__compare_in_blocks,       calc_structure_similarity),
            Comparer.CT_SSIM_WHOLE:       (self.__compare_on_whole_image,  calc_structure_similarity),
            Comparer.CT_HIST_BLOCKS:      (self.__compare_in_blocks,       calc_hist_similarity),
            Comparer.CT_HIST_WHOLE:       (self.__compare_on_whole_image,  calc_hist_similarity),
            Comparer.CT_TEMPLATE_BLOCKS:  (self.__compare_in_blocks,       calc_template_similarity),
            Comparer.CT_TEMPLATE_WHOLE:   (self.__compare_on_whole_image,  calc_template_similarity),
            Comparer.CT_AVG_HASH_BLOCKS:  (self.__compare_in_blocks,       calc_avg_hash_similarity),
            Comparer.CT_AVG_HASH_WHOLE:   (self.__compare_on_whole_image,  calc_avg_hash_similarity),
        }
        if self.compare_type not in compare_functions:
            LOGGER.error(f"Unsupported compare type {self.compare_type}.")
            raise Exception(f"Unsupported compare type {self.compare_type}.")
        
        compare_func = compare_functions.get(self.compare_type)
        
        start_time = time.monotonic_ns()
        if compare_func[1] is None:
            simlarities, mask = compare_func[0](src_frame, tgt_frame)
        else:
            simlarities, mask = compare_func[0](src_frame, tgt_frame, compare_func[1])
        end_time = time.monotonic_ns()
        LOGGER.info(f"Compare time: {(end_time - start_time) / 1e6} ms.")

        return simlarities, mask


    def __compare_with_keypoints(
            self, 
            src_frame: Frame, 
            tgt_frame: Frame
        ) -> tuple[list[CompareResult], _ImageMask]:
        """计算两个帧中所有块的特征点匹配度并返回大于阈值的块数和对应的匹配度及是否大于阈值的列表"""
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        rates, similar_count = list(), 0
        for src_block, tgt_block in zip(src_frame.frame_blocks, tgt_frame.frame_blocks):
            if src_block.has_text or tgt_block.has_text:
                similar_count += 1
                rates.append((1.0, True, src_block.block_id))
                continue
            elif src_block.descriptors is None or tgt_block.descriptors is None:
                LOGGER.warning(f"Block-{src_block.block_id} of Frame-{src_frame.frame_id} " +
                               f"or Block-{tgt_block.block_id} of Frame-{tgt_frame.frame_id} has not extracted descriptors.")
                similar_count += 1
                rates.append((1.0, True, src_block.block_id))
                continue

            matches = matcher.match(src_block.descriptors, tgt_block.descriptors)
            min_dist = min(x.distance for x in matches)

            count_good_matches = 0
            for match in matches:
                if match.distance <= max(2 * min_dist, 30):
                    count_good_matches += 1
            
            rate = count_good_matches / len(src_block.descriptors)
            rates.append(CompareResult(rate, rate >= Comparer.C_MATCHED_KEYPOINTS_THRESHOLD, src_block.block_id))
            if rate >= Comparer.C_MATCHED_KEYPOINTS_THRESHOLD:
                similar_count += 1

        if len(rates) - similar_count >= Comparer.C_UNMATCHED_BLOCKS_THRESHOLD:
            tgt_frame.may_have_exception = True
        elif src_frame.may_have_exception:
            tgt_frame.may_have_exception = True

        LOGGER.info(f"Match rates of {similar_count} blocks " +
                    f"between Frame-{src_frame.frame_id} and Frame-{tgt_frame.frame_id} is over threshold.")
        return rates, None
    

    def __compare_with_pixel_diff(self, src_frame: Frame, tgt_frame: Frame):
        """计算两个帧中所有块的像素差值并返回大于阈值的块数和对应的像素差值及是否大于阈值的列表"""
        src_image = src_frame.frame_gray.copy()
        tgt_image = tgt_frame.frame_gray.copy()
        diff_image = cv2.absdiff(src_image, tgt_image).astype(np.uint8)

        std = np.std(diff_image)
        if std > 10:
            # 可以用于检测图像视野是否发生变化
            tgt_frame.may_have_exception = True
            LOGGER.warning("Too many pixels have big difference between two images.")
            return [], np.full(src_image.shape, 255, dtype=np.uint8)         

        # flatten_diff = np.sort(diff_image.flatten())
        # slice_len = int(len(flatten_diff) * 0.01)
        # avg_diff = np.mean(flatten_diff[-slice_len:])
        avg_diff = calc_outlier_in_mask(diff_image, Comparer.C_COMPUTE_OUTLIERS) * 0.8
        if avg_diff < 10:
            # 若图像灰度值变化较大部分的变化均值较小，说明图像中可能无明显的异常物体，即图像无明显变化
            LOGGER.warning("Too few pixels have small difference between two images.")
            return [], None

        # 找到掩码边界
        old_mask = np.where(diff_image > avg_diff, 255, 0).astype(np.uint8)
        countours, _ = cv2.findContours(old_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到掩码所属图像块
        results = list(CompareResult(1.0, True, b.block_id) for b in tgt_frame.frame_blocks)
        mask = np.zeros_like(old_mask).astype(np.uint8)
        useful_contours: list[cv2.Mat] = list()
        for countour in countours:
            area = cv2.contourArea(countour)
            if area < 200:
                continue
            x1, y1, w, h = cv2.boundingRect(countour)
            x2, y2 = x1 + w, y1 + h

            useful_contours.append(countour)
            
            block_ids = block_ids_has_rect(tgt_frame.frame_width, tgt_frame.frame_height, [x1, y1, x2, y2])
            if len(block_ids) > 0.5 * len(tgt_frame.frame_blocks):
                # 若异常区域占比超过一半，则认为整个图像均有异常
                return [], np.full(src_image.shape, 255, dtype=np.uint8)
            for id in block_ids:
                results[id].similarity = 0.0
                results[id].over_thresh = False

        # 设置 countour 区域的 mask 为 255
        cv2.drawContours(mask, useful_contours, contourIdx=-1, color=255, thickness=-1)

        if len(useful_contours):
            tgt_frame.may_have_exception = True
        
        return results, mask if len(useful_contours) else None


    def __compare_in_blocks(
            self, 
            src_frame: Frame, 
            tgt_frame: Frame, 
            fit_func: Callable
        ):
        """计算两个帧中所有块的相似度并返回大于阈值的块数和对应的相似度信息列表"""
        assert len(src_frame.frame_blocks) == len(tgt_frame.frame_blocks), \
            "The number of blocks of two frames must be equal."
        
        if Comparer.C_MULTI_PROCESSING > 0:
            tasks: list[ThreadingBlocksCompareTask] = list()
            results: list[CompareResult] = list()
            slice_len = len((src_frame.frame_blocks)) // Comparer.C_MULTI_PROCESSING + 1
            for i in range(Comparer.C_MULTI_PROCESSING):
                left, right = i * slice_len, min(len(src_frame.frame_blocks), (i + 1) * slice_len)
                task = ThreadingBlocksCompareTask(
                    src_frame, tgt_frame, fit_func, left, right, Comparer.C_IMAGE_SIMILARITY_THRESHOLD
                )
                tasks.append(task)
                task.start()
            for task in tasks:
                results += task.join()
            results = sorted(results, key=lambda x: x.block_id)
        else:
            results = BlocksCompareTask.run(
                src_frame, tgt_frame, fit_func, Comparer.C_IMAGE_SIMILARITY_THRESHOLD
            )

        if Frame.F_MERGE_BLOCKS:
            results = revised_similarity_when_merging_blocks(tgt_frame, Comparer.C_IMAGE_SIMILARITY_THRESHOLD, results)

        similar_count = len(["" for s in results if s.over_thresh])
        if len(results) - similar_count >= Comparer.C_UNMATCHED_BLOCKS_THRESHOLD:
            tgt_frame.may_have_exception = True
        elif src_frame.may_have_exception:
            tgt_frame.may_have_exception = True

        LOGGER.info(f"{'-'.join(self.compare_type.split('-')[:-1])} values of {similar_count} blocks " +
                    f"between Frame-{src_frame.frame_id} and Frame-{tgt_frame.frame_id} is over threshold.")
        return results, None
        

    def __compare_on_whole_image(
            self, 
            src_frame: Frame, 
            tgt_frame: Frame, 
            fit_func: Callable
        ):
        """计算两个帧的相似度值并返回该值及是否大于阈值（对齐使用图像块进行计算的返回结果）"""
        score = float(fit_func(src_frame.frame_gray, tgt_frame.frame_gray))
        
        if score < Comparer.C_IMAGE_SIMILARITY_THRESHOLD:
            tgt_frame.may_have_exception = True
        elif src_frame.may_have_exception:
            tgt_frame.may_have_exception = True

        LOGGER.info(f"{'-'.join(self.compare_type.split('-')[:-1])} similarity value of " + 
                    f"Frame-{src_frame.frame_id} and Frame-{tgt_frame.frame_id} is {score}.")
        return [CompareResult(score, score >= Comparer.C_IMAGE_SIMILARITY_THRESHOLD)], None
