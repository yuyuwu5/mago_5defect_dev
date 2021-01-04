import torch
import torchvision
from config import *
import logging
import numpy as np
import pandas as pd
from MangoDataset import MangoDataset
from sklearn.metrics import recall_score, precision_score, classification_report

def get_classification_report(predict, real):
        predict = torch.cat(predict).cpu().numpy()
        real = torch.cat(real).cpu().numpy()
        rep = classification_report(real, predict, target_names=DEFECT_TYPE)
        rep_dict = classification_report(real, predict, target_names=DEFECT_TYPE, output_dict=True)

        return rep, rep_dict

def get_f1(clf_rep):
    ### clf_rep should be a classification report dict
    macro_precision = clf_rep['macro avg']['precision']
    macro_recall = clf_rep['macro avg']['recall']
    #print(macro_precision, macro_recall)
    f1_score = 2* macro_precision* macro_recall / (macro_precision + macro_recall)
    
    return f1_score

def getData(ratio, batch_size, seed):
    logging.info("Load Dataset")
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    #jitter_param = 0.4
    train_transform = torchvision.transforms.Compose([ \
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(15),
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

    ### TRAIN DATA ###
    train_csv = pd.read_csv(TRAIN_LABEL_CSV)
    train_img_names = train_csv.iloc[:,0].tolist()
    train_bbox = list(zip(
        train_csv.iloc[:,1].tolist(),
        train_csv.iloc[:,2].tolist(),
        train_csv.iloc[:,3].tolist(),
        train_csv.iloc[:,4].tolist(),
    ))
    train_one_hot = train_csv.iloc[:, 5:10]
    train_pos_count = train_one_hot.to_numpy().sum(axis=0)
    total_train = len(train_one_hot)
    train_pos_weight = np.array([total_train - train_pos_count[i] for i in range(5)]) / train_pos_count

    train_one_hot = train_one_hot.to_numpy().tolist()
    ### EVAL DATA ###
    eval_csv = pd.read_csv(DEV_LABEL_CSV)
    eval_img_names = eval_csv.iloc[:,0].tolist()
    eval_bbox = list(zip(
        eval_csv.iloc[:,1].tolist(),
        eval_csv.iloc[:,2].tolist(),
        eval_csv.iloc[:,3].tolist(),
        eval_csv.iloc[:,4].tolist(),
    ))
    eval_one_hot = eval_csv.iloc[:, 5:10]
    eval_one_hot = eval_one_hot.to_numpy().tolist()

    train_data = MangoDataset(TRAIN_DIR, train_img_names,bbox=train_bbox, labels=train_one_hot, transform = train_transform)
    eval_data = MangoDataset(DEV_DIR, eval_img_names, bbox=eval_bbox, labels=eval_one_hot, transform = eval_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, eval_loader, train_pos_weight