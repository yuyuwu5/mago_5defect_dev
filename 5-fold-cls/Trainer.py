import logging
import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd

from sklearn.metrics import recall_score, precision_score, classification_report
from lr_scheduler import warm_restart
from tqdm import tqdm


pd.set_option('display.max_columns', 15)

class FiveDefectTrainer(object):
    def __init__(self, train_loader, eval_loader, model, device, ckpt_path=None):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model
        self.device = device
        self.ckpt_path = ckpt_path

    def train(self, config, smooth_factor=0):
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
        logging.info("Training start")
        for e in range(config['epoch']):
            self.model.train()
            train_ans = []
            train_real_ans = []
            for x, y in tqdm(self.train_loader):
                optimizer.zero_grad()
                y_sm = y
                '''
                if smooth_factor >= 0:
                    with torch.no_grad():
                        confidence = 1 - smooth_factor
                        y_sm = torch.mul(y, confidence)
                        y_sm = torch.add(y_sm, smooth_factor*0.5)
                '''
                x, y_sm = x.to(self.device), y_sm.to(torch.float).to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y_sm)
                loss.backward()
                optimizer.step()
                scheduler.step()
                scheduler = warm_restart(scheduler, T_mult=2)
                predict_prob = torch.sigmoid(logits)
                train_ans.append(predict_prob)
                train_real_ans.append(y)
            train_f1 = self.f1_score(train_ans, train_real_ans)
            logging.info("Epoch %s, Training f1 score: %s" %(e, train_f1))
            eval_f1 = self.eval()
            logging.info("Epoch %s, Eval f1 score: %s" %(e, eval_f1))
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                logging.info("Best performance ever, save model")
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "fold%s-%.3f.pt"%(config['fold'], best_f1)))
                
        return

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
                eval_ans.append(predict_prob)
                eval_real_ans.append(y)
        eval_f1 = self.f1_score(eval_ans, eval_real_ans)
        return eval_f1


    def f1_score(self, predict, real, thresh=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        predict = torch.cat(predict).cpu().detach().numpy()
        real = torch.cat(real).cpu().numpy()
        num_class = predict.shape[1]
        f1 = {}
        f1_score = []
        r_info = {}
        p_info = {}
        for t in thresh:
            recall = []
            precision = []
            predict_t = np.where(predict>t, np.ones_like(predict), np.zeros_like(predict))
            for i in range(num_class):
                r = recall_score(real[:,i], predict_t[:,i])
                p = precision_score(real[:,i], predict_t[:,i])
                recall.append(r)
                precision.append(p)
            r_info[t] = recall
            p_info[t] = precision
            recall = np.mean(np.array(recall))
            precision = np.mean(np.array(precision))
            f1[t] = [(2*precision*recall)/(precision+recall)]
            f1_score.append((2*precision*recall)/(precision+recall))
        r_info = pd.DataFrame.from_dict(r_info, orient='index', columns=['D1', 'D2', 'D3', 'D4', 'D5'])
        r_info.index.name = "Recall"
        p_info = pd.DataFrame.from_dict(p_info, orient='index', columns=['D1', 'D2', 'D3', 'D4', 'D5'])
        p_info.index.name = "Precision"
        f1_info = pd.DataFrame.from_dict(f1)
        f1_info.index.name = "F1"
        print(r_info)
        print(p_info)
        print(f1_info)
        
        return max(f1_score)

    def test(self):
        return
