import json
import logging
import torch
import torchvision
import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from config import *
from torch.utils.tensorboard import SummaryWriter
from MangoDataset import MangoDataset
from Trainer import FiveDefectTrainer
from pytorchcv.model_provider import get_model as ptcv_get_model

from utils import getData



def main(args):
    device = torch.device("cuda:%s"%(args.cuda) if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s" %(device))
    #writer = SummaryWriter("runs/%s"%(args.description))
    ckpt_path = os.path.join(MODEL_DIR, args.description)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    
    config = {
            'lr_min': args.lr_min,
            'lr_max': args.lr_max,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'epoch': args.epoch,
            'train_ratio': args.train_ratio,
            }
    with open(os.path.join(ckpt_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    train_loader, eval_loader, pos_weight = getData(args.train_ratio, args.batch_size, args.seed)
    config['pos_weight'] = torch.tensor(pos_weight).to(torch.float).to(device)
    logging.info("Load model")
    #model = ptcv_get_model("resnext101_32x4d", pretrained=True)
    model = ptcv_get_model("resnet34", pretrained=True)
    #model = ptcv_get_model("resnet18", pretrained=True)
    #model = ptcv_get_model("resnet101", pretrained=True)
    #model = ptcv_get_model("vgg16", pretrained=True)
    ### resnet34 512, 1000
    ### resnext 2048, 1000
    model.output = torch.nn.Linear(model.output.in_features, 5)

    if args.load:
        logging.info("Continuous Training {}".format(args.load))
        model.load_state_dict(torch.load(args.load, map_location=device))
    #model.output.fc3 = torch.nn.Linear(model.output.fc3.in_features, 5)
    model.to(device)
    trainer = FiveDefectTrainer(train_loader, eval_loader, model, device, ckpt_path)
    trainer.train(config)

    return

def _parse_argument():
    parser = ArgumentParser()
    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--load', type=str, default=None)

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--seed', type=int, default=502087)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_min', type=float, default=0.)
    parser.add_argument('--lr_max', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_argument()
    main(args)

