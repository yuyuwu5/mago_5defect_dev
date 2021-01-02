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
from MangoDetection import MangoDetectionDataTest
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

def getData(bz_size):
    img_info = pd.read_csv(TEST_BOX)
    img_name = img_info['image_id'].tolist()
    img_info['w'] += img_info['x']
    img_info['h'] += img_info['y']
    x = img_info['x'].tolist()
    y = img_info['y'].tolist()
    w = img_info['w'].tolist()
    h = img_info['h'].tolist()
    bbox = list(zip(x,y,w,h))

    dataset = MangoDetectionDataTest(TEST_DIR, img_name, bbox)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = bz_size, shuffle=False, num_workers = 4, collate_fn=utils.collate_fn)
    return dataloader, img_name

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
        print("ckpt not exist")
        exit(0)

    eval_loader, img_name = getData( args.batch_size)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    num_classes = 6 #background + 5 class
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)

    predict_ans = {}
    count_predict = 0
    with torch.no_grad():
        model.eval()
        for img,attr in tqdm(eval_loader):
            img = list(im.to(device) for im in img)
            prediction = model(img)
            for p in prediction:
                one_hot = [0 for i in range(5)]
                for j in p['labels'].tolist():
                    one_hot[j-1] = 1
                predict_ans[img_name[count_predict]] = one_hot
                count_predict += 1
    predict_ans = pd.DataFrame.from_dict(predict_ans, orient='index', columns=['D1', 'D2', 'D3', 'D4', 'D5'])
    predict_ans.to_csv('predict.csv')

    return

def _parse_argument():
    parser = ArgumentParser()
    parser.add_argument('--description', type=str, required=True)

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--seed', type=int, default=502087)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
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
