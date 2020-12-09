import json
import logging
import torch
import torchvision
import os
import numpy as np

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
    jitter_param = 0.4
    train_transform = torchvision.transforms.Compose([ \
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
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
    image_data = torchvision.datasets.ImageFolder(DEFECT_DIR)
    total_len = len(image_data)
    train_len = int(ratio * total_len)
    eval_len = total_len - train_len
    train_data, eval_data = torch.utils.data.random_split(image_data, [train_len, eval_len], generator=torch.Generator().manual_seed(seed))

    train_data = MangoDataset(train_data, transform = train_transform)
    eval_data = MangoDataset(eval_data, transform = eval_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=20)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=20)

    return train_loader, eval_loader



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

    train_loader, eval_loader = getData(args.train_ratio, args.batch_size, args.seed)
    logging.info("Load model")
    model = ptcv_get_model("resnet50", pretrained=True)
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

