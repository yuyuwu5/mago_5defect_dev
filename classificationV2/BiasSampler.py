import random
from torch.utils.data.sampler import Sampler

class BiasSampler(Sampler):
    def __init__(self, pos_indice, neg_indice, pos_weight=3):
        self.pos_indice = pos_indice
        self.neg_indice = neg_indice
        self.pos_len = len(self.pos_indice)
        self.neg_len = self.pos_len * pos_weight

    def __iter__(self):
        neg = random.sample(self.neg_indice, self.neg_len)
        target = neg + self.pos_indice
        random.shuffle(target)
        return iter(target)

    def __len__(self):
        return self.pos_len + self.neg_len
