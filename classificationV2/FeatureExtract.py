import os
from tqdm import tqdm
import torch
import logging
from config import *
from utils import getData
from argparse import ArgumentParser
from pytorchcv.model_provider import get_model as ptcv_get_model

class FeatureExtract(object):
    def __init__(self, model, device, ckpt_path=None):
        #self.train_loader = train_loader
        #self.eval_loader = eval_loader
        #self.loader = loader
        self.model = model
        self.device = device
        self.ckpt_path = ckpt_path
    
    def extract(self, loader):
        print(self.model)
        self.model.to(self.device)
        self.model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for idx, (x, y) in enumerate(tqdm(loader)):
                x, y = x.to(self.device), y.to(torch.float).to(self.device)
                #print(x.shape)
                feature_x = self.model.features(x)
                feature_x = feature_x.reshape(-1, self.model.output.in_features)
                print(feature_x.shape)
                print(y.shape)
                features.append(feature_x)
                labels.append(y)

            return features, labels

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

def main(args):
    device = torch.device("cuda:%s"%(args.cuda) if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s" %(device))

    logging.info("get Data")
    train_loader, eval_loader, pos_weight = getData(args.train_ratio, args.batch_size, args.seed)

    logging.info("Load model")
    ckpt_path = args.description
    if not os.path.exists(ckpt_path):
        logging.info("CKPT not exist")
        exit(0)

    model = ptcv_get_model("resnet34", pretrained=True)
    model.output = torch.nn.Linear(model.output.in_features, 5)
    #ckpt_path = os.path.join(MODEL_DIR, args.description)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    featureExtract = FeatureExtract(model, device)
    features, labels = featureExtract.extract(train_loader)
    features = torch.cat(features).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    print(features.shape)
    print(labels.shape)
    torch.save(features, 'features/train_features512.pt')
    torch.save(labels, 'features/train_labels512.pt')

    features, labels = featureExtract.extract(eval_loader)
    features = torch.cat(features).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    print(features.shape)
    print(labels.shape)
    torch.save(features, 'features/eval_features512.pt')
    torch.save(labels, 'features/eval_labels512.pt')

if __name__ == '__main__':
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_argument()
    main(args)