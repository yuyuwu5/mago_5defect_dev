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


def getData(ratio, batch_size, seed):
    logging.info("Load Dataset")
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    #jitter_param = 0.4
    train_transform = torchvision.transforms.Compose([ \
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(15),
        #torchvision.transforms.ColorJitter(
        #    brightness=jitter_param,
        #    contrast=jitter_param,
        #    saturation=jitter_param),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),

    ])
    eval_transform = torchvision.transforms.Compose([ \
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])

    train_csv = pd.read_csv(TRAIN_CROP_CSV)
    train_img_names = train_csv.iloc[:,0].tolist()
    train_bbox = list(zip(
        train_csv.iloc[:,1].tolist(),
        train_csv.iloc[:,2].tolist(),
        train_csv.iloc[:,3].tolist(),
        train_csv.iloc[:,4].tolist(),
        ))
    train_one_hot = [[0,0,0,0,0] for i in range(len(train_img_names))]
    for i in range(len(train_img_names)):
        tmp_data = train_csv.iloc[i, 5:].dropna().tolist()
        l_tmp = len(tmp_data)
        for j in range(0, l_tmp, 5):
            disease = tmp_data[j+4]
            train_one_hot[i][DEFECT_TYPE.index(disease)] = 1
    train_pos_count = np.array(train_one_hot).sum(axis=0)
    '''
    all_prob = np.prod(train_pos_count)
    all_prob = np.array([all_prob for i in range(5)])
    train_pos_weight = all_prob / train_pos_count
    train_pos_weight = train_pos_weight / train_pos_weight[0]
    '''
    total_train = len(train_one_hot)
    train_pos_weight = np.array([total_train - train_pos_count[i] for i in range(5)]) / train_pos_count

    eval_csv = pd.read_csv(DEV_CROP_CSV)
    eval_img_names = eval_csv.iloc[:,0].tolist()
    eval_bbox = list(zip(
        eval_csv.iloc[:,1].tolist(),
        eval_csv.iloc[:,2].tolist(),
        eval_csv.iloc[:,3].tolist(),
        eval_csv.iloc[:,4].tolist(),
        ))
    eval_one_hot = [[0,0,0,0,0] for i in range(len(eval_img_names))]
    for i in range(len(eval_img_names)):
        tmp_data = eval_csv.iloc[i, 5:].dropna().tolist()
        l_tmp = len(tmp_data)
        for j in range(0, l_tmp, 5):
            disease = tmp_data[j+4]
            eval_one_hot[i][DEFECT_TYPE.index(disease)] = 1

    train_data = MangoDataset(TRAIN_DIR, train_img_names,bbox=train_bbox, labels=train_one_hot, transform = train_transform)
    eval_data = MangoDataset(DEV_DIR, eval_img_names, bbox=eval_bbox, labels=eval_one_hot, transform = eval_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, eval_loader, train_pos_weight



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
    ### resnet34 512, 1000
    ### resnext 2048, 1000
    model.output = torch.nn.Linear(model.output.in_features, 5)
    model.to(device)
    trainer = FiveDefectTrainer(train_loader, eval_loader, model, device, ckpt_path)
    trainer.train(config)

    return

def _parse_argument():
    parser = ArgumentParser()
    parser.add_argument('--description', type=str, required=True)

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--seed', type=int, default=502087)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
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

