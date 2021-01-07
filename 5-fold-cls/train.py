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
from model import EfficientModel
from lr_scheduler import WarmRestart
from ViT import VisionTransformer, CONFIGS 

def getData(batch_size, fold):
    logging.info("Load Dataset")
    #mean = (0.5, 0.5, 0.5)
    #std = (0.5, 0.5, 0.5)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    jitter_param = 0.1
    train_transform = torchvision.transforms.Compose([ \
            #torchvision.transforms.Resize((400,400)),
            #torchvision.transforms.Resize((512,512)),
            torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        #torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),

    ])
    eval_transform = torchvision.transforms.Compose([ \
            #torchvision.transforms.Resize((400,400)),
            #torchvision.transforms.Resize((512,512)),
            torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])

    all_csv = pd.read_csv(ALL_CROP_CSV)
    all_img_names = all_csv.iloc[:, 0].tolist()
    all_bbox = np.array(list(zip(
        all_csv.iloc[:,1].tolist(),
        all_csv.iloc[:,2].tolist(),
        all_csv.iloc[:,3].tolist(),
        all_csv.iloc[:,4].tolist(),
        ))
        )

    if not os.path.exists(ALL_ONE_HOT):
        all_one_hot = [[0,0,0,0,0] for i in range(COUNT_TOTAL_IMG)]
        for i in range(COUNT_TOTAL_IMG):
            tmp_data = all_csv.iloc[i, 5:].dropna().tolist()
            l_tmp = len(tmp_data)
            for j in range(0, l_tmp, 5):
                disease = tmp_data[j+4]
                all_one_hot[i][DEFECT_TYPE.index(disease)] = 1
        all_one_hot = np.array(all_one_hot)
        np.save(ALL_ONE_HOT, all_one_hot)
    else:
        all_one_hot = np.load(ALL_ONE_HOT)

    train_idx = []
    for f in range(FOLD_NUM):
        if f != fold:
            train_idx += FOLD[f]
    eval_idx = FOLD[fold]

    train_img_names = np.array(all_img_names)[train_idx].tolist()
    train_one_hot = all_one_hot[train_idx,:]
    train_bbox = all_bbox[train_idx].tolist()

    eval_img_names = np.array(all_img_names)[eval_idx].tolist()
    eval_one_hot = all_one_hot[eval_idx,:]
    eval_bbox = all_bbox[eval_idx].tolist()
    
    train_pos_count = train_one_hot.sum(axis=0)
    total_train = len(train_one_hot)
    train_pos_weight = np.array([total_train-train_pos_count[i] for i in range(5)]) / train_pos_count 

    train_data = MangoDataset(ALL_DIR, train_img_names,bbox=train_bbox, labels=train_one_hot, transform = train_transform)
    eval_data = MangoDataset(ALL_DIR, eval_img_names, bbox=eval_bbox, labels=eval_one_hot, transform = eval_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, eval_loader, train_pos_weight


def main(args):
    device = torch.device("cuda:%s"%(args.cuda) if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s" %(device))
    logging.info("Fold %s"%(args.fold))
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
            'fold': args.fold,
            }
    
    train_loader, eval_loader, pos_weight = getData(args.batch_size, args.fold)
    config['pos_weight'] = torch.tensor(pos_weight).to(torch.float).to(device)
                
    logging.info("Load model")
    
    #model = ptcv_get_model("efficientnet_b4", pretrained=True)
    #model = ptcv_get_model("resnet152", pretrained=True)
    #model = ptcv_get_model("resnet18", pretrained=True)
    model = ptcv_get_model("seresnext50_32x4d", pretrained=True)
    #print(model)
    #model.output = torch.nn.Linear(51200, 5) #resnet18
    #model.output = torch.nn.Linear(204800, 5)
    model.output = torch.nn.Linear(model.output.in_features, 5)
    #print(model)
    #model = EfficientModel(5)
    #vi_config = CONFIGS['ViT-B_16']
    #model = VisionTransformer(vi_config, num_classes=1000, zero_head=False, img_size=224, vis=False)
    #model.load_from(np.load('../model/ViT-B_16.npz'))
    #model.head = torch.nn.Linear(768, 5)

    config['optimizer'] = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #config['optimizer'] = torch.optim.SGD(model.parameters(), lr=0.001, momentum=config['momentum'], weight_decay=config['weight_decay'])
    config['scheduler'] = WarmRestart(config['optimizer'], T_max=10, T_mult=1, eta_min=1e-5)
    model.to(device)
    trainer = FiveDefectTrainer(train_loader, eval_loader, model, device, ckpt_path)
    trainer.train(config, smooth_factor=args.sm)

    return

def _parse_argument():
    parser = ArgumentParser()
    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3,4])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--seed', type=int, default=502087)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--sm', type=float, default=0.1)
    parser.add_argument('--cuda', type=int, default=0)
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

