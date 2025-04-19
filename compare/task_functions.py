# coding: utf-8
import threading
from typing import Callable

from skimage.metrics import structural_similarity as compare_ssim

from frame.frame import Frame
from compare.result import CompareResult


class ThreadingBlocksCompareTask(threading.Thread):
    def __init__(
            self, src_frame: Frame, tgt_frame: Frame, fit_func: Callable,
            left: int = None, right: int = None, thresh: float = 0.9
        ):
        super().__init__()
        self.src_frame = src_frame
        self.tgt_frame = tgt_frame
        self.fit_func = fit_func
        self.left = left if left is not None else 0
        self.right = right if right is not None else len(src_frame.frame_blocks)
        self.thresh = thresh
        self.results = list()


    def run(self):
        for src_block, tgt_block in zip(
                self.src_frame.frame_blocks[self.left: self.right], 
                self.tgt_frame.frame_blocks[self.left: self.right]
            ):
            src_block_image = self.src_frame.frame_gray[
                src_block.top: src_block.top + src_block.height, 
                src_block.left: src_block.left + src_block.width
            ]
            tgt_block_image = self.tgt_frame.frame_gray[
                tgt_block.top: tgt_block.top + tgt_block.height,
                tgt_block.left: tgt_block.left + tgt_block.width
            ]
            
            if src_block.has_text or tgt_block.has_text:
                ssim = 1.0
            else:
                ssim = float(self.fit_func(src_block_image, tgt_block_image))

            self.results.append(CompareResult(ssim, ssim >= self.thresh, src_block.block_id))


    def join(self):
        super().join()
        return self.results
    
    
class BlocksCompareTask(object):
    @staticmethod
    def run(
            src_frame: Frame, tgt_frame: Frame,
            fit_func: Callable = compare_ssim, thresh: float = 0.9
        ) -> list[CompareResult]:
        results = list()
        for src_block, tgt_block in zip(src_frame.frame_blocks, tgt_frame.frame_blocks):
            src_block_image = src_frame.frame_gray[
                src_block.top: src_block.top + src_block.height, 
                src_block.left: src_block.left + src_block.width
            ]
            tgt_block_image = tgt_frame.frame_gray[
                tgt_block.top: tgt_block.top + tgt_block.height,
                tgt_block.left: tgt_block.left + tgt_block.width
            ]
            
            if src_block.has_text or tgt_block.has_text:
                score = 1.0
            else:
                score = float(fit_func(src_block_image, tgt_block_image))

            results.append(CompareResult(score, score >= thresh, src_block.block_id))
        
        return results
