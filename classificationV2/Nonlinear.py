import os
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from argparse import ArgumentParser
import torch.nn.functional as F
from utils import getData, get_f1, get_classification_report

class linearRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num):
        super(linearRegression, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, class_num)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out

class linearRegressionTrainer():
    def __init__(self, train_loader, eval_loader, model, device):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model
        self.device = device

    def train(self, config):
        logging.info("Start Training")
        criterion = nn.BCEWithLogitsLoss(pos_weight=config['pos_weight'])
        if "optimizer" in config:
            optimizer = config["optimizer"]
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=config['momentum'], weight_decay=config['weight_decay'])

        lr_steps = config['epoch']*len(self.train_loader)
        if "scheduler" in config:
            scheduler = config["scheduler"]
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps//2, lr_steps*3//4], gamma=0.1)

        best_f1 = 0.
        for e in range(config['epoch']):
            train_ans = []
            train_real_ans = []
            self.model.train()
            for idx, (x, y) in enumerate(tqdm(self.train_loader)):
                optimizer.zero_grad()
                x, y = x.to(self.device), y.to(torch.float).to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                predict_prob = torch.sigmoid(logits)
                train_ans.append(torch.where(predict_prob>0.5, torch.ones_like(logits), torch.zeros_like(logits)))
                train_real_ans.append(y)

            train_rep, train_rep_dict = get_classification_report(train_ans, train_real_ans)
            train_f1 = get_f1(train_rep_dict)
            logging.info("Epoch %s, Training:\n%s\n F1: %s" %(e, train_rep, train_f1))

            eval_ans, eval_real_ans = self.eval()
            eval_rep, eval_rep_dict = get_classification_report(eval_ans, eval_real_ans)
            eval_f1 = get_f1(eval_rep_dict)
            logging.info("Epoch %s, Eval:\n%s\n F1: %s" %(e, eval_rep, eval_f1))
    
    def eval(self):
        logging.info("Start Evaluation")
        eval_ans = []
        eval_real_ans = []
        self.model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(tqdm(self.eval_loader)):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                predict_prob = torch.sigmoid(logits)
                eval_ans.append(torch.where(predict_prob>0.5, torch.ones_like(logits), torch.zeros_like(logits)))
                eval_real_ans.append(y)
        
        #eval_f1 = self.f1_score(eval_ans, eval_real_ans)
        return eval_ans, eval_real_ans

def main(args):
    device = torch.device("cuda:%s"%(args.cuda) if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s" %(device))

    train_features = torch.load('features/train_features512.pt')
    train_labels = torch.load('features/train_labels512.pt')
    train_features = torch.tensor(train_features)
    train_labels = torch.tensor(train_labels)

    eval_features = torch.load('features/eval_features512.pt')
    eval_labels = torch.load('features/eval_labels512.pt')
    eval_features = torch.tensor(eval_features)
    eval_labels = torch.tensor(eval_labels)

    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    eval_dataset = torch.utils.data.TensorDataset(eval_features, eval_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    _, _, pos_weight = getData(args.train_ratio, args.batch_size, args.seed)
    config = {
        "epoch": args.epoch,
        'lr_min': args.lr_min,
        'lr_max': args.lr_max,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'train_ratio': args.train_ratio,
        "hidden_dim":128,
        "pos_weight":torch.tensor(pos_weight).to(torch.float).to(device)
    }
    model = linearRegression(train_features.shape[1], config["hidden_dim"], 5)
    model.to(device)
    trainer = linearRegressionTrainer(train_loader, eval_loader, model, device)
    trainer.train(config)

    #for idx, (x, y) in enumerate(tqdm(train_loader)):
    #    print(x.shape)

def _parse_argument():
    parser = ArgumentParser()
    #parser.add_argument('--description', type=str, required=True)
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