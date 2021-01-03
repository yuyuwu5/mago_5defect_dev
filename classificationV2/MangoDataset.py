import os
import torch

from PIL import Image
from torch.utils.data import Dataset

class MangoDataset(Dataset):
    def __init__(self, data_root, img_names, bbox=None, labels=None, transform=None):
        self.data_root = data_root
        self.img_names = img_names
        self.bbox = bbox
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.img_names[idx])
        img = Image.open(img_path)
        if self.bbox and self.bbox[idx]!=[0,0,0,0]:
            img = img.crop(self.bbox[idx])
        y = []
        if self.transform:
            img = self.transform(img)
        if self.labels:
            y = torch.tensor(self.labels[idx])
        return img, y
    
    def __len__(self):
        return len(self.img_names)
