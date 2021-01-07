import torch
import torch.nn as nn

from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

class EfficientModel(nn.Module):
    def __init__(self, num_class):
        super(EfficientModel, self).__init__()
        self.effnet = EfficientNet.from_pretrained("efficientnet-b3")                
        #self.effnet = EfficientNet.from_pretrained("efficientnet-b5")                
        #self.effnet = EfficientNet.from_pretrained("efficientnet-b7")                
        #self.effnet._conv_stem.in_channels = 1
        #weight = self.effnet._conv_stem.weight.mean(1, keepdim=True)
        #self.effnet._conv_stem.weight = torch.nn.Parameter(weight)                                   
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(1536, num_class)                                                        
        ##self.out = nn.Linear(2048, num_class)                                                        
        #self.out = nn.Linear(2560, num_class)  #v7                                             

    def forward(self, image):
        batch_size, _, _, _ = image.shape
        x = self.effnet.extract_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        outputs = self.out(self.dropout(x))
        return outputs
