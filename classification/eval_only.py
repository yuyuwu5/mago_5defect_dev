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


#TEST_CKPT = "tmp/0.6542193743051171.pt"
TEST_CKPT = "tmp/0.6538884055411468.pt"


def getData(ratio, batch_size):
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

    image_data = torchvision.datasets.ImageFolder(DEV_DEFECT_DIR)
    test_data = MangoDataset(image_data, transform = eval_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=20)
    return test_loader, None
    #image_data = torchvision.datasets.ImageFolder(TRAIN_DEFECT_DIR)

    '''
    total_len = len(image_data)
    train_len = int(ratio * total_len)
    eval_len = total_len - train_len
    train_data, eval_data = torch.utils.data.random_split(image_data, [train_len, eval_len])
    
    train_data = MangoDataset(train_data, transform = train_transform)
    eval_data = MangoDataset(eval_data, transform = eval_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=20)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=20)

    return train_loader, eval_loader
    '''


def main(args):
    device = torch.device("cuda:%s"%(args.cuda) if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s" %(device))
    

    test_loader, eval_loader = getData(args.train_ratio, args.batch_size)
    logging.info("Load model")
    model = ptcv_get_model("resnet50", pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, MODEL_DIR, TEST_CKPT)))

    trainer = FiveDefectTrainer(eval_loader, test_loader, model, device, )
    f1 = trainer.eval()
    print("f1: ", f1)


    return

def _parse_argument():
    parser = ArgumentParser()

    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_argument()
    main(args)

