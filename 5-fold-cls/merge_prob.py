import torch
import pandas as pd
import numpy as np

name_list = pd.read_csv("col1.csv")['image_id'].tolist()
prob_list = [
        '../model/n_resnet34_sm0.1_thresh0.65/0.559.pt.logit',
        '../model/resnet34_sm0.1_thresh0.65/0.554.pt.logit',
        #'../model/resnet152/0.524.pt.logit',
        #'../model/efficient/0.528.pt.logit',
        #'../model/efficient_512/0.535.pt.logit',
        #'../model/efficient-b7-sgd-512/0.538.pt.logit',
        #'../model/efficient-b7-400/0.532.pt.logit',
        #'../model/seresnext/0.526.pt.logit',
        '../model/resnet18/0.552.pt.logit',
        ]


p = None
for a in prob_list:
    prob = torch.load(a)
    if p is not None:
        p += prob
    else:
        p = prob
p = p / len(prob_list)
p = torch.where(p>0.6, torch.ones_like(p), torch.zeros_like(p))
result = p.numpy()
out = pd.DataFrame(data=result, index=name_list, columns=["D1", "D2", "D3", "D4", "D5"])
out.index.name="image_id"
out.to_csv("ensemble.csv")
