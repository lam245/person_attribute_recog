import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle

from dataset.augmentation import get_transform
from dataset.multi_label.coco import COCO14
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_factory import build_backbone, build_classifier
from torchvision import transforms

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict

from tools.function import get_model_log_path, get_reload_weight,get_reload_weight_test
from tools.utils import set_seed, str2bool, time_str
from models.backbone import swin_transformer, resnet, bninception
# from models.backbone.tresnet import tresnet
from losses import bceloss, scaledbceloss

from PIL import Image

set_seed(605)

def main(cfg, args):
    # Define the image transform
    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Build the model
    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)
    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=26,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale=cfg.CLASSIFIER.SCALE
    )
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Load the trained weights
    model = get_reload_weight_test("/exp_result/PA100k/resnet50.base.adam/img_model", model,"/ckpt_max_2024-08-24_04:33:15.pth")
    model.eval()

    # Process the image
    img_path = args.image_path
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    # Inference
    with torch.no_grad():
        logits, _ = model(img_tensor)
        probs = torch.sigmoid(logits[0])

    # Print results
    print(probs)

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--cfg", help="decide which cfg to use", type=str, required=True)
    parser.add_argument("--image_path", help="path to the image for inference", type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)
    main(cfg, args)