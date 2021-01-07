import os
import torch
import random

from PIL import Image,ImageFilter,ImageDraw
from torch.utils.data import Dataset
from config import *
import torchvision

class MangoDataset(Dataset):
    def __init__(self, data_root, img_names, bbox=None, labels=None, transform=None, train=True):
        self.data_root = data_root
        self.img_names = img_names
        self.bbox = bbox
        self.train = train
        self.labels = labels
        self.transform = transform
        #self.milk = os.listdir(DEFECT_PATCH[0])
        self.mechanical_damage = os.listdir(DEFECT_PATCH[1])
        #self.black_spot = DEFECT_PATCH[4]

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.img_names[idx])
        img = Image.open(img_path)
        if self.bbox and self.bbox[idx]!=[0,0,0,0]:
            img = img.crop(self.bbox[idx])
        y = []
        if self.labels is not None:
            y = torch.tensor(self.labels[idx])

        if self.train and random.random() < 0.3:
            patch_name = os.path.join(DEFECT_PATCH[1], random.choice(self.mechanical_damage))
            patch = Image.open(patch_name)
            #patch = Image.new("L", patch.size, 0)
            #draw = ImageDraw.Draw(patch)
            #draw.rectangle((140, 50, 260, 170), fill=255)
            if y[1]==0 and patch.size[0] < img.size[0] and patch.size[1] < img.size[1]:
                mask_im = Image.new("L", patch.size, 0)
                #patch = torchvision.transforms.RandomAffine(45, scale=(0.5,2),shear=10)(patch)
                patch = torchvision.transforms.RandomHorizontalFlip()(patch)
                patch = torchvision.transforms.RandomVerticalFlip()(patch)
                draw = ImageDraw.Draw(mask_im)
                draw.ellipse((0,0,patch.size[0], patch.size[1]),fill=255)
                mask_im_blur = mask_im.filter(ImageFilter.GaussianBlur(5))
                img.paste(patch, (random.randint(int(1/3* img.size[0]), int(2/3* img.size[0])), random.randint(int(1/3* img.size[1]), int(2/3* img.size[1]))), mask_im_blur)
                y[1] = 1

        if self.transform:
            img = self.transform(img)
        return img, y
    
    def __len__(self):
        return len(self.img_names)
