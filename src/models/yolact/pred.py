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

import torch
import torch.backends.cudnn as cudnn
import cv2

CUDA = False

def predImage(net, image):

    if CUDA:
        frame = torch.from_numpy(image).cuda().float()
    else:
        frame = torch.from_numpy(image).float()
    batch = FastBaseTransform(CUDA)(frame.unsqueeze(0))
    preds = net(batch)
    pred = preds[0]
    del pred['net'],
    non_tensor_pred = {entry:pred['detection'][entry].tolist() for entry in ['box', 'mask', 'class', 'score',]}# 'proto']}
    non_tensor_pred['class'] = [trash_can.class_names[c] for c in non_tensor_pred['class']]
    return non_tensor_pred


def getModel(trained_model_path = "models/yolact/best_model.pth", config_path = "yolact_resnet50_config"):
    with torch.no_grad():
        print("setting config")
        set_cfg(config_path)
        print("setting tensor")
        if CUDA:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        print("creating network")
        net = Yolact()
        print("loading weghts")
        net.load_weights(trained_model_path, CUDA)
        print("eval")
        net.eval()
        print("cuding network")
        if CUDA:
            net =net.cuda()
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        return net


    # evalimage("weights/yolact_resnet50_26_53333.pth", cv2.imread("../../../data/TrashCan/train/vid_000003_frame0000011.jpg"))


