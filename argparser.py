# coding: utf-8
import os
import argparse
import warnings

import yaml
import torch

from frame.frame import Frame, FrameBlock
from compare.compare import Comparer
from videostream.stream import ChinaMobileStream
from videostream.stream2image import Stream2Image
from detect import Detector
from logger import LOGGER
from window import SlideWindow
from detect import BaseDetector


warnings.filterwarnings("ignore")


class ArgParser(object):
    def __init__(self, yaml_path: str = "./config.yaml"):
        parser = argparse.ArgumentParser()
        self.__add_arguments(parser)

        self.__args: argparse.Namespace = parser.parse_args()

        # assert self.__args.yaml_path is not None, 'Please set the path to yaml file'
        yaml_path = self.__args.yaml_path if self.__args.yaml_path is not None else yaml_path
        
        self.arguments: dict[str, int | float | str] = yaml.safe_load(open(yaml_path, 'r'))

        self.__clear_arguments()

        self.__print_args()

    
    def __print_args(self):
        print("=" * 50)
        for key, value in self.arguments.items():
            print(f"{' ' * (25 - len(key))}{key}: {value}")
        print("=" * 50)

    
    def __clear_arguments(self):
        if type(self.arguments["labels-map"]) == str:
            fp = open(self.arguments["labels-map"], 'r')
            if fp is None:
                LOGGER.error(f"Labels file {self.arguments['labels-map']} does not exist")
                raise FileNotFoundError(f"{self.arguments['labels-map']} does not exist")
            
            labels = fp.readlines()
            LOGGER.info(f"Read labels from file: {self.arguments['labels-map']}")
            labels = list(map(lambda s: s.strip(), labels))
            fp.close()
            self.arguments["labels-map"] = labels
        
        if self.arguments["gpu-device"].lower() != "cpu":
            if not torch.cuda.is_available():
                LOGGER.warning("GPU/CUDA is not available, using CPU instead")
                self.arguments["gpu-device"] = "cpu"

        if self.arguments["detect-text"]:
            LOGGER.info("Detect text with OCR, OCR is only running on CPU so far.")

        if self.arguments["results-present"] == 1:
            LOGGER.info("Results are only shown on time.")
        elif self.arguments["results-present"] == 2:
            LOGGER.info("Results are only saved to ./results.")
            if not os.path.exists("./results/"):
                os.mkdir("./results/")
        elif self.arguments["results-present"] == 3:
            LOGGER.info("Results are saved and shown on time.")
            if not os.path.exists("./results/"):
                os.mkdir("./results/")
        else:
            LOGGER.error("Invalid results present option.")
            raise ValueError("Invalid results present option.")
        
        if not os.path.exists(self.arguments["web-images-path"]):
            os.mkdir(self.arguments["web-images-path"])

        if not self.arguments["user-info-file"].endswith(".json"):
            LOGGER.warning(f"User info file {self.arguments['user-info-file']} should be a json file")

        if not isinstance(self.arguments["compare-type"], str) and \
                not isinstance(self.arguments["compare-type"], list):
            LOGGER.warning(f"Compare type {self.arguments['compare-type']} is invalid, use 'PIXEL-DIFF' instead.")
            self.arguments["compare-type"] = "PIXEL-DIFF"


    @staticmethod
    def __add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('-yaml', '--yaml-path', type=str, default=None, help='path to the configuration file (yaml)')


class ArgSetter(object):
    @staticmethod
    def set_args(arguments: dict = None):
        if arguments is None:
            LOGGER.error("No configuration read yet.")
            raise NotImplementedError("No configuration read yet.")

        Frame.F_STRIP_RATE_INV         = arguments["strip-rate"]
        Frame.F_SPLIT_RATE_INV         = arguments["split-rate"]
        Frame.F_MERGE_BLOCKS           = arguments["merge-blocks"]
        Frame.F_KEYPOINT_TYPE          = arguments["keypoint-type"]
        Frame.F_KEYPOINT_NUM_PER_BLOCK = arguments["keypoint-num"]
        Frame.F_DETECT_TEXT_IN_BLOCK   = arguments["detect-text"]
        # Frame.F_OCR_WITH_GPU           = arguments["gpu-device"].lower() != "cpu"
        Frame.F_RESULTS_PRESENT        = arguments["results-present"]

        if isinstance(arguments["compare-type"], str):
            FrameBlock.FB_CAL_KPS_DESC = arguments["compare-type"] == Comparer.CT_KEY_POINTS
        else:  # type is list
            FrameBlock.FB_CAL_KPS_DESC = Comparer.CT_KEY_POINTS in arguments["compare-type"]

        Comparer.C_IMAGE_SIMILARITY_THRESHOLD   = arguments["image-similarity-thresh"]
        Comparer.C_UNMATCHED_BLOCKS_THRESHOLD   = arguments["unmatched-blocks-thresh"]
        Comparer.C_MATCHED_KEYPOINTS_THRESHOLD  = arguments["matched-keypoints-thresh"]
        Comparer.C_CHECK_VIEW_CHANGED           = arguments["check-view-changed"]
        Comparer.C_MULTI_PROCESSING             = arguments["multi-processing-for-blocks"]
        Comparer.C_COMPUTE_OUTLIERS             = arguments["outliers-compute"]

        ChinaMobileStream.S_USER_INFO_JSON      = arguments["user-info-file"]

        Stream2Image.SI_ROOT_SAVE_DIRECTORY     = arguments["web-images-path"]
        Stream2Image.SI_SAMPLING_GAP_MINUTE     = arguments["stream-sampling-gap"]
        
        Detector.DT_IMAGE_PROCESS_DEVICE = arguments["gpu-device"]

        BaseDetector.D_CONFIDENCE_THRESHOLD     = arguments["confidence-thresh"]
        BaseDetector.D_USE_CONFIDENCE_THRESHOLD = arguments["use-confidence-thresh"]
        BaseDetector.D_NMS_IOU_THRESHOLD        = arguments["nms-iou-thresh"]
        BaseDetector.D_LABELS_MAP               = dict((i, val) for i, val in enumerate(arguments["labels-map"]))
        BaseDetector.D_WARM_UP_TIMES            = arguments["warmup-times"]

        SlideWindow.SLIDE_WINDOW_SIZE = arguments["window-size"]
