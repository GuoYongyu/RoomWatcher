# coding: utf-8
import numpy as np
from paddleocr import PaddleOCR

from logger import LOGGER


OCR_DETECTOR = PaddleOCR(use_angle_cls=True, use_gpu=False, show_log=False)


def find_text_boxes(frame: np.ndarray, ocr: PaddleOCR) -> list:
    assert isinstance(frame, np.ndarray), "frame must be a numpy array"
    height, _, _ = frame.shape

    # 距离图像边界上下 10% 的区域作为检测区域
    ratio = 0.1

    ocr_results = ocr.ocr(img=frame)
    det_results = list()
    for res in ocr_results:
        x1, y1 = res[0][0][0]
        x2, y2 = res[0][0][1]
        x3, y3 = res[0][0][2]
        x4, y4 = res[0][0][3]

        # 计算框两条边的斜率
        k1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float("inf")
        k2 = (y3 - y4) / (x3 - x4) if (x3 - x4) != 0 else float("inf")
        # 如果斜率离水平线很远，则不考虑检测框
        if abs(k1) >= 0.1 and abs(k2) >= 0.1:
            continue
        
        new_x1, new_y1 = int(min(x1, x2, x3, x4)), int(min(y1, y2, y3, y4))
        new_x2, new_y2 = int(max(x1, x2, x3, x4)), int(max(y1, y2, y3, y4))

        # 检测框高度占比超过图像高度的 15%，则不考虑检测框
        if abs(new_y2 - new_y1) / height > 0.15:
            continue

        # 如果检测框不在图像边界上下 10% 的区域内，则不考虑检测框
        if ratio * height <= new_y1 and new_y2 <= height - ratio * height:
            continue
        
        det_results.append([new_x1, new_y1, new_x2, new_y2])

    return det_results


def area_of_intersected_rects(
        rect1: list[int],  # [x_l, y_t, width, height]
        rect2: list[int],  # [x_l, y_t, width, height]
        O_loc: str = ("left-top", "left-bottom")  # location of coordinate origin
    ) -> int:
    # 计算两个矩形框相交部分的面积
    r1_x1, r1_y1, r1_w, r1_h = rect1
    r2_x1, r2_y1, r2_w, r2_h = rect2

    # 改为左下和右上坐标
    r1_y2 = r1_y1
    r1_y1 += r1_h
    r1_x2 = r1_x1 + r1_w
    r2_y2 = r2_y1
    r2_y1 += r2_h
    r2_x2 = r2_x1 + r2_w

    if O_loc == "left-top":
        inter_x1 = max(r1_x1, r2_x1)
        inter_y1 = min(r1_y1, r2_y1)
        inter_x2 = min(r1_x2, r2_x2)
        inter_y2 = max(r1_y2, r2_y2)
        # 检测相交是否成立
        if inter_x1 >= inter_x2 or inter_y1 <= inter_y2:
            return 0
    elif O_loc == "left-bottom":
        inter_x1 = max(r1_x1, r2_x1)
        inter_y1 = max(r1_y1, r2_y1)
        inter_x2 = min(r1_x2, r2_x2)
        inter_y2 = min(r1_y2, r2_y2)
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0
    else:
        LOGGER.error(f"Location of coordinate origin '{O_loc}' not supported!")
        raise ValueError("Location of coordinate origin should be 'left-top' or 'left-bottom'")

    inter_width = abs(inter_x2 - inter_x1)
    inter_height = abs(inter_y2 - inter_y1)
    return int(inter_width * inter_height)


def block_with_text(
        text_boxes: list[list[int]],  # [x_l, y_t, x_r, y_b]
        block_rect: list[int],        # [x_l, y_t, width, height]
        threshold: float = 0.25
    ) -> bool:
    # 图像块是否包含文本
    _, _, width, height = block_rect
    block_area = width * height

    for box in text_boxes:
        tx1, ty1, tx2, ty2 = box

        inter_area = area_of_intersected_rects(
            block_rect, [tx1, ty1, tx2 - tx1, ty2 - ty1], "left-top"
        )

        # 相交面积占图像块面积的比例
        iou = inter_area / block_area
        if iou > threshold:
            return True

    return False


def merge_intersecting_blocks(block_rects: list) -> tuple[list[float], list[list[int]]]:
    def is_intersect(rect1: tuple | list, rect2: tuple | list):
            x1, y1, x2, y2 = rect1
            x3, y3, x4, y4 = rect2
            w1, h1 = x2 - x1, y2 - y1
            w2, h2 = x4 - x3, y4 - y3
            w = abs((x1 + x2) / 2 - (x3 + x4) / 2)
            h = abs((y1 + y2) / 2 - (y3 + y4) / 2)
            return 2 * w <= w1 + w2 + 2 and 2 * h <= h1 + h2 + 2

    def is_in_cluster(rect: list, clusters: list[list]) -> int:
        if len(clusters) == 0:
            return -1
        for c_id in range(len(clusters)):
            for c_rect in clusters[c_id]:
                if is_intersect(rect, c_rect):
                    return c_id
        return -1

    def intersecting_rectangle_clusters(
            rectangles: list[list[int]]
        ) -> list[list[int]]:
        """
            Return:
                - 返回相交矩形（图像块）所构成的簇
            Params:
                - block_rects: 图像块的矩形坐标列表，每个矩形坐标由四个整数组成，分别表示左上角和右下角的横纵坐标
        """
        clusters = list()
        for rect in rectangles:
            c_id = is_in_cluster(rect, clusters)
            if c_id < 0:
                cluster = [rect]
                clusters.append(cluster)
            else:
                clusters[c_id].append(rect)
        return clusters

    def merge_rectangles_in_cluster(cluster: list[list]):
        rectangles = list()
        for cluster in clusters:
            x1, y1, x2, y2 = cluster[0]
            for rect in cluster[1:]:
                x3, y3, x4, y4 = rect
                x1 = min(x1, x3)
                y1 = min(y1, y3)
                x2 = max(x2, x4)
                y2 = max(y2, y4)
            rectangles.append([x1, y1, x2, y2])
        return rectangles

    clusters = intersecting_rectangle_clusters(block_rects)
    return merge_rectangles_in_cluster(clusters)
