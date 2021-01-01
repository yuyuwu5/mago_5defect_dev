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

def getData(data_path, img_dir, bz_size, device, mode="train"):
    train_data = pd.read_csv(data_path, header=None)
    ans = {}
    bounding_box = []
    labels = []
    img_name = []
    for idx in range(train_data.shape[0]):
        data = train_data.iloc[idx].dropna().tolist()
        img_name.append(os.path.join(img_dir, data[0]))
        box = []
        lab = []
        one_hot = [0 for i in range(5)]
        for j in range(1, len(data), 5):
            x, y, dx, dy, disease = data[j:j+5]
            x, y, dx, dy = int(x), int(y), int(dx), int(dy)
            box.append([x, y, x+dx, y+dy])
            lab.append(DEFECT_TYPE.index(disease)+1)
            one_hot[DEFECT_TYPE.index(disease)] = 1
        ans[idx] = one_hot
        bounding_box.append(box)
        labels.append(lab)

    dataset = MangoDetectionData(img_name, bounding_box, labels, device, transforms=getTransform(mode=='train'))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = bz_size, shuffle=mode=='train', num_workers = 4, collate_fn = utils.collate_fn)
    ans = pd.DataFrame.from_dict(ans, orient='index', columns=['D1', 'D2', 'D3', 'D4', 'D5'])
    return dataloader, ans

def getTransform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def f1Score(predict_frame, ans_frame):
    precision = []
    recall = []
    for c in predict_frame.columns:
        y_pred = predict_frame[c].tolist()
        y_true = ans_frame[c].tolist()
        precision.append(
                precision_score(y_true, y_pred))
        recall.append(
                recall_score(y_true, y_pred))
    precision_ma = np.array(precision).mean()
    recall_ma = np.array(recall).mean()
    f1 = 2*recall_ma*precision_ma / (recall_ma+precision_ma)
    return f1

def main(args):
    device = torch.device("cuda:%s"%(args.cuda) if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s" %(device))
    #writer = SummaryWriter("runs/%s"%(args.description))
    ckpt_path = os.path.join(MODEL_DIR, args.description)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)                        
    config = {
            'lr_max': args.lr_max,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'epoch': args.epoch,
            'train_ratio': args.train_ratio,
            }
    with open(os.path.join(ckpt_path, 'config.json'), 'w') as f:
        json.dump(config, f)


    train_loader, train_ans = getData(TRAIN_CSV, TRAIN_DIR, args.batch_size, device, mode = "train")
    eval_loader, eval_ans = getData(DEV_CSV, DEV_DIR, 8, device, mode = "eval")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    num_classes = 6 #background + 5 class
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = args.lr_max, momentum = args.momentum, weight_decay = args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_f1 = 0
    for e in range(args.epoch):
        model.train()
        train_one_epoch(model, optimizer, train_loader, device, e, print_freq=100)
        lr_scheduler.step()
        #evaluate(model, eval_loader, device=device)
        predict_ans = {}
        with torch.no_grad():
            model.eval()
            for img, attr in tqdm(eval_loader):
                img = list(im.to(device) for im in img)
                prediction = model(img)
                for a, p in zip(attr, prediction):
                    one_hot = [0 for i in range(5)]
                    img_id = a['image_id'].tolist()[0]
                    for j in p['labels'].tolist():
                        one_hot[j-1] = 1
                    predict_ans[img_id] = one_hot
        predict_ans = pd.DataFrame.from_dict(predict_ans, orient='index', columns=['D1', 'D2', 'D3', 'D4', 'D5'])
        f1 = f1Score(predict_ans, eval_ans)
        print("Epoch %s, f1= %s" %(e, f1))
        if f1 > best_f1:
            best_f1 = f1
            logging.info("Best performance ever, save model")
            torch.save(model.state_dict(), os.path.join(ckpt_path, "%.4f.pt"%(best_f1)))
                                                                                      





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
