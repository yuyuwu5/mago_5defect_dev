import logging
import pandas as pd
import numpy as np
import torch
import torchvision
import transforms as T
import utils
import json

from argparse import ArgumentParser
from config import *
from engine import train_one_epoch, evaluate
from MangoDetection import MangoDetectionData
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

def getData(bz_size):
    train_data = pd.read_csv(TEST_BOX)
    img_name = train_data['image_id'].tolist()
    train_data['w'] += train_data['x']
    train_data['h'] += train_data['y']
    x = train_data['x'].tolist()
    y = train_data['y'].tolist()
    w = train_data['w'].tolist()
    h = train_data['h'].tolist()
    bbox = list(zip(x,y,w,h))
    bounding_box = [[b] for b in bbox]
    
    labels = [[1] for i in img_name]
    dataset = MangoDetectionData(TEST_DIR, img_name, bounding_box, labels, transforms=getTransform(True))
    dataset_test = MangoDetectionData(TEST_DIR, img_name, bounding_box, labels, transforms=getTransform(False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-1000])
    dataset_test = torch.utils.data.Subset(dataset, indices[-1000:])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = bz_size, shuffle=True, num_workers = 4, collate_fn = utils.collate_fn)
    dataloader_eval = torch.utils.data.DataLoader(dataset_test, batch_size = 4, shuffle=False, num_workers = 4, collate_fn = utils.collate_fn)
    return dataloader, dataloader_eval

def getTransform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):
    device = torch.device("cuda:%s"%(args.cuda) if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s" %(device))
    #writer = SummaryWriter("runs/%s"%(args.description))

    ckpt_path = os.path.join(MODEL_DIR, args.description)
    if not os.path.exists(ckpt_path):
        print("Path not exist")
        exit(0)

    train_loader, eval_loader = getData(args.batch_size )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    num_classes = 2 #background + mango
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)

    best_f1 = 0
    model.eval()
    with torch.no_grad():
       for img, attr in eval_loader:
           img = list(im.to(device) for im in img)
           prediction = model(img)
           print(prediction)
           exit(0)


                                                                
    return

def _parse_argument():
    parser = ArgumentParser()
    parser.add_argument('--description', type=str, required=True)

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--seed', type=int, default=502087)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_min', type=float, default=0.)
    parser.add_argument('--lr_max', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_argument()
    main(args)
