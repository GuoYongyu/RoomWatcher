# coding: utf-8
import math
from typing import TypeAlias

import cv2
import numpy as np

from frame.frame import Frame, FrameBlock
from compare.result import CompareResult
from logger import LOGGER


MaskLike: TypeAlias = np.ndarray


def calc_hist_similarity(src_image: np.ndarray, tgt_image: np.ndarray) -> float:
    """计算两张图像的直方图的相似度"""
    src_hist = cv2.calcHist([src_image], [0], None, [256], [0.0, 255.0])
    tgt_hist = cv2.calcHist([tgt_image], [0], None, [256], [0.0, 255.0])
    src_hist = cv2.normalize(src_hist, src_hist).flatten()
    tgt_hist = cv2.normalize(tgt_hist, tgt_hist).flatten()
    
    sim = cv2.compareHist(src_hist, tgt_hist, cv2.HISTCMP_INTERSECT)
    return sim


def calc_template_similarity(src_image: np.ndarray, tgt_image: np.ndarray) -> float:
    """计算两张图像模板匹配的相似度"""
    result = cv2.matchTemplate(src_image, tgt_image, cv2.TM_CCORR_NORMED)
    max_val, _, _, _ = cv2.minMaxLoc(result)
    return max_val


def calc_avg_hash_similarity(src_image: np.ndarray, tgt_image: np.ndarray) -> float:
    """计算两张图像的平均哈希值的相似度"""
    src_hash = np.where(src_image > np.mean(src_image), 1, 0)
    tgt_hash = np.where(tgt_image > np.mean(tgt_image), 1, 0)
    distance = np.count_nonzero(src_hash != tgt_hash)
    return 1 - (distance / src_hash.size)


def check_view_changed(
        cmp_results: list[CompareResult], 
        thresh: float = 0.8,
        mask: MaskLike = None
    ) -> bool:
    """检查视野是否发生变化"""
    if np.mean(mask) > 128:
        return True

    if len(cmp_results) == 0:
        return False
    
    assert isinstance(cmp_results, list) \
           and isinstance(cmp_results[0], CompareResult), "No Compare Results"
    
    if cmp_results[0].block_id == -1:  # compare on whole image
        return cmp_results[0].similarity < thresh
    else:
        count = 0
        for r in cmp_results:
            if r.similarity < thresh:
                count += 1
        return count > math.sqrt(len(cmp_results)) + 0.5


def revise_blocks_with_exceptions(
        last_block_ids: list[int], 
        curr_block_ids: list[int],
    ) -> list[int]:
    """
        - 保留当前一场图像块与上一次比较的图像块不同的部分
        - 如果 mask 不为 None，则会通过 block id 对 mask 进行过滤
    """
    id1, id2 = 0, 0
    block_ids = list()

    while id1 < len(last_block_ids) and id2 < len(curr_block_ids):
        if last_block_ids[id1] == curr_block_ids[id2]:
            id1 += 1
            id2 += 1
        else:
            block_ids.append(curr_block_ids[id2])
            id2 += 1
    
    while id2 < len(curr_block_ids):
        block_ids.append(curr_block_ids[id2])
        id2 += 1

    return block_ids


def is_rects_intersect(rect1: tuple | list, rect2: tuple | list):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    w1, h1 = x2 - x1, y2 - y1
    w2, h2 = x4 - x3, y4 - y3
    w = abs((x1 + x2) / 2 - (x3 + x4) / 2)
    h = abs((y1 + y2) / 2 - (y3 + y4) / 2)
    return 2 * w <= w1 + w2 + 2 and 2 * h <= h1 + h2 + 2


def intersecting_rectangle_clusters(
        block_rects: list[list[int]], block_ids: list[int]
    ) -> list[list[int]]:
    """
        Return:
            - 返回相交矩形（图像块）所构成的簇
            - 返回一个二维列表，每个簇是一个列表，列表中的元素是图像块的索引
        Params:
            - block_rects: 图像块的矩形坐标列表，每个矩形坐标由四个整数组成，分别表示左上角和右下角的横纵坐标
            - block_ids: 图像块的索引列表
    """
    assert len(block_rects) == len(block_ids)
    
    def is_in_cluster(rect: list, clusters: list[list]) -> int:
        if len(clusters) == 0:
            return -1
        for c_id in range(len(clusters)):
            for id in clusters[c_id]:
                if is_rects_intersect(rect, blocks_map[id]):
                    return c_id
        return -1
    
    clusters: list[list[int]] = list()
    blocks_map = dict((id, block) for id, block in zip(block_ids, block_rects))
    for rect, id in zip(block_rects, block_ids):
        c_id = is_in_cluster(rect, clusters)
        if c_id < 0:
            cluster = [id]
            clusters.append(cluster)
        else:
            clusters[c_id].append(id)
    return clusters


def revised_similarity_when_merging_blocks(
        frame: Frame, thresh: float,
        block_sim_infos: list[CompareResult],
    ) -> list[CompareResult]:
    """
        - 如果需要合并图像块，则重新细化每个图像块的相似度
        - 细化方式为对异常图像块簇计算平均相似度并赋予这些图像块
        - 对于重新计算了相似度的图像块，再次跟阈值 thresh 比较
    """
    block_ids = dict((info.block_id, i) for i, info in enumerate(block_sim_infos) if not info.over_thresh)
    block_rects = list()
    for id in block_ids.keys():
        b = frame.frame_blocks[id]
        block_rects.append([b.left, b.top, b.left + b.width, b.top + b.height])
    
    clusters = intersecting_rectangle_clusters(block_rects, block_ids)
    for cluster in clusters:
        avg_sim = sum([block_sim_infos[id].similarity for id in cluster]) / len(cluster)
        for id in cluster:
            block_sim_infos[block_ids[id]] = CompareResult(avg_sim, avg_sim >= thresh, id)
    
    return block_sim_infos


def remove_intersecting_rectangles(
        new_rects: list[list[int]], old_rects: list[list[int]]
    ) -> list[list[int]]:
    rectangles = list()
    for n_rect in new_rects:
        flag = False
        for o_rect in old_rects:
            if is_rects_intersect(n_rect, o_rect):
                flag = True
                break
        if flag:
            rectangles.append(n_rect)
    return rectangles


def block_ids_has_rect(frame_width: int, frame_height: int, rect: list[int]) -> list[int]:
    """找到 rect 所覆盖的所有图像块的 id"""
    width_step, height_step = \
        frame_width // Frame.F_STRIP_RATE_INV, frame_height // Frame.F_STRIP_RATE_INV
    rx1, ry1, rx2, ry2 = rect
    assert rx1 < rx2 and ry1 < ry2, "Invalid Rectangle"

    col_min, row_min = rx1 // width_step, ry1 // height_step
    col_max = min(Frame.F_STRIP_RATE_INV - 2, rx2 // width_step)
    row_max = min(Frame.F_STRIP_RATE_INV - 2, ry2 // height_step)
    
    block_ids = list()
    for col in range(col_min, col_max + 1):
        for row in range(row_min, row_max + 1):
            block_ids.append(row * (Frame.F_STRIP_RATE_INV - 1) + col)
    return block_ids


def clear_mask(mask: MaskLike, frame: Frame, block_ids: list[int]) -> MaskLike:
    """清除部分图像块的 mask"""
    block_rects: list[list[int]] = list()
    for id in block_ids:
        b: FrameBlock = frame.frame_blocks[id]
        block_rects.append([b.left, b.top, b.left + b.width, b.top + b.height])
    clusters: list[list[int]] = intersecting_rectangle_clusters(block_rects, block_ids)

    for cluster in clusters:
        x1, y1, x2, y2 = frame.frame_width, frame.frame_height, 0, 0
        for id in cluster:
            x1 = min(x1, frame.frame_blocks[id].left)
            y1 = min(y1, frame.frame_blocks[id].top)
            x2 = max(x2, frame.frame_blocks[id].left + frame.frame_blocks[id].width)
            y2 = max(y2, frame.frame_blocks[id].top + frame.frame_blocks[id].height)
        mask[x1: x2, y1: y2] = 0x00
    
    return mask


def calc_outlier_in_mask(
        mask: MaskLike, 
        algorithm: str = ["mean", "z-score", "iqr", "ransac", "sort"][0]
    ) -> float:
    """找到 mask 中的异常点平均值"""

    def __calc_mean(mask: MaskLike) -> float:
        flatten_diff = np.sort(mask.flatten())
        slice_len = int(len(flatten_diff) * 0.01)
        avg_diff = np.mean(flatten_diff[-slice_len:])
        return avg_diff
    
    def __calc_z_score(mask: MaskLike) -> float:
        mean = np.mean(mask)
        std = np.std(mask)
        z_scores = np.abs((mask - mean) / std)
        outliers = mask[z_scores > 3]
        return np.mean(outliers)
    
    def __calc_iqr(mask: MaskLike) -> float:
        q1 = np.percentile(mask, 25)
        q3 = np.percentile(mask, 75)
        iqr = q3 - q1
        upper_bound = q3 + 5.0 * iqr
        outliers = mask[mask > upper_bound]
        return np.mean(outliers)
    
    def __calc_ransac(mask: MaskLike) -> float:
        mask_flatten = np.sort(mask.flatten())[::-1]  # descending order
        mean = np.mean(mask_flatten[0: 500])
        it = 0
        for i in range(500, mask_flatten.shape[0], 500):
            tmp_mean = np.mean(mask_flatten[i: i + 500])
            if tmp_mean / mean < 0.75:
                break
            mean = np.mean(mask_flatten[0: i + 500])
            if i + 500 >= mask_flatten.shape[0] * 0.01:
                break   
            it += 1
            if it >= 500:
                break
        return mean
    
    def __calc_sort(mask: MaskLike) -> float:
        flatten_diff = np.sort(mask.flatten())
        it = 0
        for i in range(500, len(flatten_diff), 500):
            if flatten_diff[i] / flatten_diff[i - 500] < 0.75:
                break
            it += 1
            if it >= 500:
                break
        return np.mean(flatten_diff[i]) if i < len(flatten_diff) and it < 500 else 0

    if algorithm.lower() == "mean":
        return __calc_mean(mask)
    elif algorithm.lower() == "z-score":
        return __calc_z_score(mask)
    elif algorithm.lower() == "iqr":
        return __calc_iqr(mask)
    elif algorithm.lower() == "ransac":
        return __calc_ransac(mask)
    elif algorithm.lower() == "sort":
        return __calc_sort(mask)
    else:
        raise ValueError(f"Unknown algorithm type '{algorithm.upper()}' when computing mask's outliers.")
