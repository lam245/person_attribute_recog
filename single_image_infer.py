import argparse
import os
import pickle
from PIL import Image
from tqdm import tqdm
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from dataset.augmentation import get_transform
from dataset.multi_label.coco import COCO14
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_factory import build_backbone, build_classifier



from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict

from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from models.backbone import swin_transformer, resnet, bninception
# from models.backbone.tresnet import tresnet
from losses import bceloss, scaledbceloss
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
set_seed(605)

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name

def get_reload_weight(model_path, model):
    load_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    if isinstance(load_dict, OrderedDict):
        pretrain_dict = load_dict
    else:
        pretrain_dict = load_dict['state_dicts']
        print(f"best performance {load_dict['metric']} in epoch : {load_dict['epoch']}")
    model.load_state_dict(pretrain_dict, strict=False)
    return model

def save_results(results, output_file, file_format):
    if file_format == 'txt':
        with open(output_file, 'w') as f:
            for img_name, attributes in results.items():
                f.write(f"{img_name}: {', '.join(attributes)}\n")
    elif file_format == 'csv':
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Attributes'])
            for img_name, attributes in results.items():
                writer.writerow([img_name, ', '.join(attributes)])
    print(f"Results saved to {output_file}")

def main(cfg, args):
    # Define the image transform
    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset info
    dataset_info = pickle.load(open("data/PA100k/dataset_all.pkl", 'rb+'))
    attr_id = dataset_info.attr_name

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
    model = get_reload_weight("exp_result/PA100k/resnet50.base.adam/img_model/ckpt_max_2024-08-24_04:33:15.pth", model)
    model.eval()

    # Create dataset and dataloader
    dataset = ImageFolderDataset(args.image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Inference
    results = {}
    with torch.no_grad():
        for batch, img_names in tqdm(dataloader, desc="Processing images"):
            if torch.cuda.is_available():
                batch = batch.cuda()
            
            logits, _ = model(batch)
            
            # Handle the case where logits is a list
            if isinstance(logits, list):
                logits = logits[0]  # Assume the first element is the main output
            
            probs = torch.sigmoid(logits)
            pred_labels = probs > 0.8

            for i, img_name in enumerate(img_names):
                result = [attr for attr, pred in zip(attr_id, pred_labels[i]) if pred]
                results[img_name] = result

    # Save results
    save_results(results, args.output_file, args.output_format)

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--cfg", help="decide which cfg to use", type=str, required=True)
    parser.add_argument("--image_folder", help="path to the folder containing images for inference", type=str, required=True)
    parser.add_argument("--batch_size", help="batch size for inference", type=int, default=32)
    parser.add_argument("--output_file", help="path to save the inference results", type=str, default="inference_results.txt")
    parser.add_argument("--output_format", help="format of the output file (txt or csv)", type=str, choices=['txt', 'csv'], default='txt')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)
    main(cfg, args)