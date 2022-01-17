from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset, trash_can

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def evalimage(weights_path:str, image):
    net = Yolact()
    net.load_weights(weights_path)
    net.eval()
    net =net.cuda()
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    cfg.mask_proto_debug = False

    frame = torch.from_numpy(image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    pred = preds[0]
    del pred['net'],
    non_tensor_pred = {entry:pred['detection'][entry].tolist() for entry in ['box', 'mask', 'class', 'score',]}# 'proto']}
    non_tensor_pred['class'] = [trash_can.class_names[c] for c in non_tensor_pred['class']]
    return non_tensor_pred


trained_model = "weights/yolact_resnet50_26_53333.pth"
with torch.no_grad():
    model_path = SavePath.from_str(trained_model)
    set_cfg(model_path.model_name + '_config')
    cudnn.fastest = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    evalimage("weights/yolact_resnet50_26_53333.pth", cv2.imread("../../../data/TrashCan/train/vid_000003_frame0000011.jpg"))


