import pandas as pd
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from config import *
from PIL import Image


class MangoDataset(Dataset):
    def __init__(self, data_list, label, mode="train"):
        self.label = label
        self.data_list = data_list
        
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
        ])

        if mode == "train":
            self.img_root = TRAIN_DIR
            '''
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
                ])
            '''
        elif mode == "dev":
            self.img_root = DEV_DIR
            '''
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
                ])
            '''
        else:
            print("mode error")
            exit(0)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.data_list[idx])
        img = Image.open(img_path)
        img = self.transforms(img)
        return img, self.label[idx]
    
    def __len__(self):
        return len(self.data_list)
