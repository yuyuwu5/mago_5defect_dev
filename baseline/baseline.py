import torch
import torch.nn as nn
import torchvision
import logging
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from config import *
from MangoDataset import MangoDataset
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, classification_report
from sklearn.feature_selection import SelectPercentile, f_classif


def parseLabel(data):
    num = len(data)
    all_label = []
    all_name = []
    for idx in range(num):
        label = [0 for i in range(5)]
        d = data.iloc[idx].dropna().tolist()
        img_name = d[0]
        for j in range(1, len(d), 5):
            x, y, dx, dy, disease = d[j:j+5]
            label[DEFECT_TYPE.index(disease)] = 1
        all_name.append(img_name)
        all_label.append(label)
    return all_name, all_label


def prepareData():
    train_data = pd.read_csv(TRAIN_CSV, header=None)
    train_name, train_y = parseLabel(train_data)
    train_set = MangoDataset(train_name, train_y)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle=False)

    eval_data = pd.read_csv(DEV_CSV, header=None)
    eval_name, eval_y = parseLabel(eval_data)
    eval_set = MangoDataset(eval_name, eval_y, mode="dev")
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size = 128, shuffle=False)

    return train_loader, eval_loader, np.array(train_y), np.array(eval_y)

def featureExtraction():
    logging.info("Feature extraction")
    device = torch.device("cuda:%s"%(args.cuda) if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s" %(device))

    logging.info("Parse data and build dataset")
    train_loader, eval_loader = prepareData()

    logging.info("Load alexnet and vgg16")
    alexnet = torchvision.models.alexnet(pretrained=True)
    alexnet.eval()
    alexnet.to(device)

    vggnet = torchvision.models.vgg16(pretrained=True)
    vggnet.eval()
    vggnet.to(device)

    logging.info("Start feature extraction")
    train_feature = []
    with torch.no_grad():
        for x,y in tqdm(train_loader):
            x = x.to(device)
            alex_feature = alexnet(x)
            vgg_feature = vggnet(x)
            full_feature = torch.cat([alex_feature, vgg_feature], axis = 1)
            train_feature.append(full_feature)
    train_feature = torch.cat(train_feature)
    torch.save(train_feature, TRAIN_FEATURE_PATH)
    
    eval_feature = []
    with torch.no_grad():
        for x,y in tqdm(eval_loader):
            x = x.to(device)
            alex_feature = alexnet(x)
            vgg_feature = vggnet(x)
            full_feature = torch.cat([alex_feature, vgg_feature], axis = 1)
            eval_feature.append(full_feature)
    eval_feature = torch.cat(eval_feature)
    torch.save(eval_feature, EVAL_FEATURE_PATH)
    return train_feature, eval_feature


def main(args):
    if not os.path.exists(TRAIN_FEATURE_PATH) or not os.path.exists(EVAL_FEATURE_PATH):
        train_x, eval_x = featureExtraction()
    else:
        train_x = torch.load(TRAIN_FEATURE_PATH, map_location=torch.device('cpu'))
        eval_x = torch.load(EVAL_FEATURE_PATH, map_location=torch.device('cpu'))
    _, _, train_y, eval_y = prepareData()

    total_c = [1.0, 0.1, 0.1,  0.1,1.0]
    total_p = [50, 10, 20, 10, 20]

    #sample_train = np.random.randint(train_x.shape[0], size=train_x.shape[0]//3)

    precisions = []
    recalls = []

    for idx, (p,c) in enumerate(zip(total_p, total_c)):
        sample_train_n = (train_y[:,idx] == 1).nonzero()[0]
        sample_support = (train_y[:,idx] == 0).nonzero()[0]
        pos_num = len(sample_train_n)
        sample_support = np.random.choice(sample_support, size=pos_num)
        sample_train = np.concatenate([sample_train_n, sample_support])

        clf = Pipeline([
            ('anova', SelectPercentile(f_classif)),
            ('svc', SVC(C=c, kernel="linear", class_weight='balanced', verbose=False))
            ])
        clf.set_params(anova__percentile=p)
        clf.fit(train_x[sample_train], train_y[sample_train, idx])
        y_pred = clf.predict(eval_x)
        
        print(DEFECT_TYPE[idx])
        print(classification_report(eval_y[:,idx], y_pred))

        recall = recall_score(eval_y[:,idx], y_pred)
        print("recall: ", recall)
        recalls.append(recall)

        precision = precision_score(eval_y[:,idx], y_pred)
        print("precision: ", precision)
        precisions.append(precision)

    final_precision = np.mean(np.array(precisions))
    final_recall = np.mean(np.array(recalls))
    f1 = (2*final_precision*final_recall)/(final_precision+final_recall)
    print("f1: ", f1)

    return 

def _parse_argument():
    parser = ArgumentParser()

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_argument()
    main(args)

