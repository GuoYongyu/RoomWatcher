# coding: utf-8
import torch
from torch import nn

from faster_rcnn.resnet50 import Resnet50RoIHead
from faster_rcnn.resnet50 import get_features_classes_in_resnet50
from faster_rcnn.rpn import RegionProposalNetwork


class FasterRCNN(nn.Module):
    def __init__(self,  
            num_classes: int,  
            mode: str = "predict",
            feat_stride: int = 16,
            anchor_scales: list = [8, 16, 32],
            ratios: list = [0.5, 1, 2],
            # backbone: str = 'resnet',
            # pretrained: bool = False
        ):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        self.extractor, classifier = get_features_classes_in_resnet50()
        # construct classifier net
        self.rpn = RegionProposalNetwork(
            1024, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            mode=mode
        )
        # construct classifier net
        self.head = Resnet50RoIHead(
            n_class=num_classes + 1,
            roi_size=14,
            spatial_scale=1,
            classifier=classifier
        )
            
            
    def forward(self, 
                x: torch.Tensor, 
                scale: float = 1., 
                mode: str = "forward"
        ):
        if mode == "forward":
            img_size = x.shape[2:]
            base_feature = self.extractor.forward(x)

            # get ROI boxes
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            # get classifier results and regression results
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            base_feature = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            # get ROI boxes
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            # get classifier results and regression results
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
