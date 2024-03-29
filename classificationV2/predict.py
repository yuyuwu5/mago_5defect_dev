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
from tqdm import tqdm
from model import EfficientModel


def getData(ratio, batch_size, seed):
    logging.info("Load Dataset")
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    #jitter_param = 0.4
    eval_transform = torchvision.transforms.Compose([ \
            #torchvision.transforms.Resize((224,224)),
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])


    eval_csv = pd.read_csv(TEST_BOX)
    eval_img_names = eval_csv.iloc[:,0].tolist()

    eval_csv.iloc[:,3] += eval_csv.iloc[:,1]
    eval_csv.iloc[:,4] += eval_csv.iloc[:,2]

    eval_bbox = list(zip(
        eval_csv.iloc[:,1].tolist(),
        eval_csv.iloc[:,2].tolist(),
        eval_csv.iloc[:,3].tolist(),
        eval_csv.iloc[:,4].tolist(),
        ))

    eval_data = MangoDataset(TEST_DIR, eval_img_names, bbox=eval_bbox,  transform = eval_transform)

    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=4)
    return eval_loader, eval_img_names


def main(args):
    device = torch.device("cuda:%s"%(args.cuda) if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s" %(device))
    #writer = SummaryWriter("runs/%s"%(args.description))
    ckpt_path = os.path.join(MODEL_DIR, args.description)
    if not os.path.exists(ckpt_path):
        logging.info("CKPT not exist")
        exit(0)
    
    eval_loader, name_list = getData(args.train_ratio, args.batch_size, args.seed)
    logging.info("Load model")
    '''
    model = ptcv_get_model("resnext101_32x4d", pretrained=True)
    model.output = torch.nn.Linear(2048, 5)
    '''
    model = EfficientModel(1)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    test_ans = []
    logging.info("Start inference")
    model.eval()
    with torch.no_grad():
        for x,y in tqdm(eval_loader):
            x = x.to(device)
            logits = model(x)
            predict_prob = torch.sigmoid(logits)
            test_ans.append(torch.where(predict_prob>0.5, torch.ones_like(logits), torch.zeros_like(logits)))
    result = torch.cat(test_ans).cpu().numpy()
    out = pd.DataFrame(data=result, index=name_list)#, columns=["D1","D2","D3", "D4","D5"])
    #out = pd.DataFrame(data=result, index=name_list, columns=["D1","D2","D3", "D4","D5"])
    out.index.name = "image_id"
    out.to_csv('test_predict.csv')
    logging.info("Done")

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

