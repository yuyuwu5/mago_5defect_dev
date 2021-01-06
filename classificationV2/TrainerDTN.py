import logging
import numpy as np
import os
import torch
import torch.nn as nn

from sklearn.metrics import recall_score, precision_score, classification_report
from tqdm import tqdm
from config import DEFECT_TYPE
from FocalLoss import FocalLoss2d

from generatorNet import GeneratorNet

class DTNTrainer(object):
    def __init__(self, train_loader, gen_loader1, gen_loader2, eval_loader, model, device, ckpt_path=None):
        self.train_loader = train_loader
        self.gen_loader1 = gen_loader1
        self.gen_loader2 = gen_loader2

        self.eval_loader = eval_loader
        self.model = model
        self.device = device
        self.ckpt_path = ckpt_path

        self.model_G = GeneratorNet(num_classes=5, dim=512, norm=True, scale=True).to(self.device)

    def train(self, config):
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
            for idx, ZIP in enumerate(tqdm(zip(self.train_loader, self.gen_loader1, self.gen_loader2))):
                optimizer.zero_grad()

                train_sample, gen_sample1, gen_sample2 = ZIP
                train_samplex, train_sampley = train_sample
                gen_sample1x, gen_sample1y = gen_sample1
                gen_sample2x, gen_sample2y = gen_sample2
                
                train_feature = self.model.features(train_samplex.to(self.device)).squeeze()
                gen_feature1 = self.model.features(gen_sample1x.to(self.device)).squeeze()
                gen_feature2 = self.model.features(gen_sample2x.to(self.device)).squeeze()
                #print(train_feature.shape)
                #print(gen_feature1.shape)
                #print(gen_feature2.shape)
                gen_feature, _ = self.model_G(gen_feature1, gen_feature2, train_feature)
                #print(gen_feature.shape)
                rebuild_features = torch.cat([train_feature.unsqueeze(1), gen_feature], axis=1)
                rebuild_features = rebuild_features.view(-1, rebuild_features.shape[2])
                #rebuild_features = rebuild_features.mean(dim=1)
                #print(rebuild_features.shape)
                train_sampley = train_sampley.unsqueeze(1).repeat(1, gen_feature1.shape[0]+1, 1)
                
                train_sampley = train_sampley.view(-1, train_sampley.shape[2])
                
                #x, y = x.to(self.device), y.to(torch.float).to(self.device)
                
                logits = self.model.output(rebuild_features)
                loss = criterion(logits, train_sampley.to(torch.float).to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                predict_prob = torch.sigmoid(logits)
                train_ans.append(torch.where(predict_prob>0.5, torch.ones_like(logits), torch.zeros_like(logits)))
                train_real_ans.append(train_sampley.to(torch.float).to(self.device))
                
            #train_f1 = self.f1_score(train_ans, train_real_ans)
            #logging.info("Epoch %s, Training f1 score: %s" %(e, train_f1))
            train_rep, train_rep_dict = self.get_classification_report(train_ans, train_real_ans)
            train_f1 = self.get_f1(train_rep_dict)
            logging.info("Epoch %s, Training:\n%s\n F1: %s" %(e, train_rep, train_f1))

            eval_ans, eval_real_ans = self.eval()
            eval_rep, eval_rep_dict = self.get_classification_report(eval_ans, eval_real_ans)
            eval_f1 = self.get_f1(eval_rep_dict)
            logging.info("Epoch %s, Eval:\n%s\n F1: %s" %(e, eval_rep, eval_f1))

            if eval_f1 > best_f1:
                best_f1 = eval_f1
                logging.info("Best performance ever, save model")
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "%.3f.pt"%(best_f1)))
                
        return

    def eval(self):
        logging.info("Start Evaluation")
        eval_ans = []
        eval_real_ans = []
        self.model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(tqdm(self.eval_loader)):
                #bs, ncrop, c, m, n = x.shape
                #x = x.reshape(-1, c, m, n)
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                #logits = logits.reshape(bs, ncrop, -1)
                #logits = logits.mean(axis=1)
                
                predict_prob = torch.sigmoid(logits)
                eval_ans.append(torch.where(predict_prob>0.5, torch.ones_like(logits), torch.zeros_like(logits)))
                eval_real_ans.append(y)
        
        #eval_f1 = self.f1_score(eval_ans, eval_real_ans)
        return eval_ans, eval_real_ans

    def get_classification_report(self, predict, real):
        predict = torch.cat(predict).cpu().numpy()
        real = torch.cat(real).cpu().numpy()
        rep = classification_report(real, predict, target_names=DEFECT_TYPE)
        rep_dict = classification_report(real, predict, target_names=DEFECT_TYPE, output_dict=True)

        return rep, rep_dict

    def get_f1(self, clf_rep):
        ### clf_rep should be a classification report dict
        macro_precision = clf_rep['macro avg']['precision']
        macro_recall = clf_rep['macro avg']['recall']
        #print(macro_precision, macro_recall)
        f1_score = 2* macro_precision* macro_recall / (macro_precision + macro_recall)
        
        return f1_score

    def f1_score(self, predict, real):
        predict = torch.cat(predict).cpu().numpy()
        real = torch.cat(real).cpu().numpy()
        recall = []
        precision = []
        for i in range(5):
            print(real[i].shape)
            print(predict[i].shape)
            #exit(0)
            r = recall_score(real[i], predict[i])
            p = precision_score(real[i], predict[i])
            recall.append(r)
            precision.append(p)
        logging.info("Five class recall: %s, %s, %s, %s, %s", recall[0], recall[1], recall[2], recall[3], recall[4])
        logging.info("Five class precision: %s, %s, %s, %s, %s", precision[0], precision[1], precision[2], precision[3], precision[4])
        recall = np.mean(np.array(recall))
        precision = np.mean(np.array(precision))
        return (2*precision*recall)/(precision+recall)

    def test(self):
        return
