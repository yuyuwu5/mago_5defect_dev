import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class MangoDetectionData(Dataset):
    def __init__(self, img_list, defect_pos_list, defect_y_list, device, transforms = None, ):
        self.img_list = img_list 
        self.defect_pos_list = defect_pos_list
        self.defect_y_list = defect_y_list
        self.transforms = transforms
        return
        
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('RGB')
        boxes = torch.as_tensor(self.defect_pos_list[idx], dtype=torch.float32)
        label = torch.tensor(self.defect_y_list[idx], dtype=torch.int64)
        img_id = torch.tensor([idx])
        area = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        iscrowd = torch.zeros((len(self.defect_y_list[idx])), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "label": label,
            "image_id": img_id,
            "area": area,
            "iscrowd": iscrowd,
            "labels": label,
                }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_list)
