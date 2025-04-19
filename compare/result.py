# coding: utf-8


class CompareResult(object):
    def __init__(
            self, 
            sim_score: float = 0.9999, 
            over_thresh: bool = True, 
            block_idx: int = -1  # -1 means comparing on whole image
        ):
        self.similarity   = sim_score
        self.over_thresh  = over_thresh
        self.block_id     = block_idx 
