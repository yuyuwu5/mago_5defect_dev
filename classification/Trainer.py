import logging
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, precision_score, classification_report

def tmptransform(x):
    xx = []
    for i in x:
        if i == 0:
            xx.append(1)
        elif i == 1:
            xx.append(3)
        elif i == 2:
            xx.append(2)
        elif i == 3:
            xx.append(0)
        elif i == 4:
            xx.append(4)
    return xx


class FiveDefectTrainer(object):
    def __init__(self, train_loader, eval_loader, model, device, ckpt_path=None):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model
        self.device = device
        self.ckpt_path = ckpt_path

    def train(self, config):
        if "criterion" in config:
            criterion = config["criterion"]
        else:
            criterion = nn.CrossEntropyLoss()
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
            for idx, (x, y) in enumerate(self.train_loader):
                optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_ans += torch.argmax(logits, axis=1).tolist()
                train_real_ans += y.tolist()
            train_f1 = self.f1_score(train_ans, train_real_ans)
            logging.info("Epoch %s, Training f1 score: %s" %(e, train_f1))
            eval_f1 = self.eval()
            logging.info("Epoch %s, Eval f1 score: %s" %(e, eval_f1))
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                logging.info("Best performance ever, save model")
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "%.4f.pt"%(best_f1)))
                
        return

    def eval(self):
        eval_ans = []
        eval_real_ans = []
        self.model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.eval_loader):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                eval_ans += torch.argmax(logits, axis=1).tolist()
                eval_real_ans += y.tolist()
        eval_f1 = self.f1_score(eval_ans, eval_real_ans)
        return eval_f1

    def precision(self, predict, real, class_label):
        total_num = 0
        correct = 0
        for pre, lab in zip(predict, real):
            if pre == class_label:
                total_num += 1
                if pre == lab:
                    correct += 1
        return correct / total_num if total_num > 0 else 0

    def recall(self, predict, real, class_label):
        total_num = 0
        correct = 0
        for pre, lab in zip(predict, real):
            if lab == class_label:
                total_num += 1
                if pre == lab:
                    correct += 1
        return correct / total_num if total_num > 0 else 0

    def f1_score(self, predict, real):
        print(classification_report(real, predict))
        recall = recall_score(real, predict, average=None)
        precision = precision_score(real, predict, average=None)
        recall = np.mean(np.array(recall))
        precision = np.mean(np.array(precision))
        return (2*precision*recall)/(precision+recall)
        '''
        class_precision = []
        class_recall = []
        for i in range(5):
            class_precision.append(self.precision(predict, real, i))
            class_recall.append(self.recall(predict, real, i))
        avg_precision = np.array(class_precision).mean()
        avg_recall = np.array(class_recall).mean()
        f1 = (2 * avg_recall * avg_precision) / (avg_recall + avg_precision)
        return f1
        '''
        

    def test(self):
        return
