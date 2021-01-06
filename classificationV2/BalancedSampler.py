import torch
import numpy as np
from utils import getData
from torch.utils.data.sampler import BatchSampler, Sampler

class BalancedSampler(Sampler):
    def __init__(self, dataset, n_classes=5, n_samples=20000):
        self.n_samples = n_samples
        self.n_classes = n_classes

        self.dataset = dataset
        self.labels = np.array(self.dataset.labels)
        self.img_names = self.dataset.img_names

        self.classes_base =[
            np.where(self.labels[:, i]==1)[0] for i in range(n_classes)
        ]
        self.classes_single =[
            np.where((self.labels[:,i]==1) & (self.labels.sum(axis=1)==1))[0] for i in range(n_classes)
        ]
        self.num_each_class = [len(class_) for class_ in self.classes_base]
        

    def __iter__(self):
        #print(self.classes_base)
        epoch_list = np.arange(self.labels.shape[0])
        """
        for idx, c in enumerate(self.classes_single):
            #print([c.shape ])
            if idx == 2:
                print(self.labels[c])
        exit(0)
        for class_idx in range(len(self.classes_base)):
            class_ = self.classes_base[class_idx]
            np.random.shuffle(class_)
            choiced_data = np.random.choice(class_, max(self.num_each_class))
            #print(choiced_data.shape)
            epoch_list.extend(choiced_data.tolist())
        """
        machine_harms = self.classes_single[1]
        black_dots = self.classes_single[4]
        aug_machine_harms = np.random.choice(machine_harms, self.num_each_class[1]* 5)
        aug_black_dots = np.random.choice(black_dots, self.num_each_class[4]* 5)
        #print(aug_machine_harms.shape)
        #print(aug_black_dots.shape)
        #print(epoch_list.shape)
        epoch_list = np.concatenate([epoch_list, aug_machine_harms, aug_black_dots], axis=0)
        #print(epoch_list.shape)
        #exit(0)

        np.random.shuffle(epoch_list)

        return iter(epoch_list)
    
    def __len__(self):
        return len(self.dataset) + 5* (self.num_each_class[1] + self.num_each_class[4])

if __name__ == '__main__':
    train_ratio = 0.8
    batch_size = 4
    seed = 502087
    train_data, eval_data, pos_weight = getData(train_ratio, batch_size, seed)
    sampler = BalancedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=4)
    #print(next(iter(sampler)))
    all_labels = []
    for idx, (imgs, labels) in enumerate(train_loader):
        #print(labels.shape)
        all_labels.append(labels)
        
        all_labels_tensor = torch.cat(all_labels)
        if idx % 10 == 0:
            print(idx)
        print(all_labels_tensor.sum(axis=0))
        print(all_labels_tensor.sum())