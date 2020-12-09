from torch.utils.data import Dataset

class MangoDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img = self.dataset[index][0]

        if self.transform:
            x = self.transform(img)
        else:
            x = img
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)
