# coding: utf-8
### FIXME: DEPRECATED FILE
## dependencies

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy,Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint



from data import *

## Constants
PLACE_PULSE_PATH ='votes_clean.csv'
IMAGES_PATH= 'placepulse/'
MODEL_PATH = 'model.pth'

#SsCnn definition

class SsCnn(nn.Module):

    def __init__(self, model, finetune=False):
        super(SsCnn, self).__init__()
        self.cnn = model(pretrained=True).features
        x = torch.randn([3,224,224]).unsqueeze(0)
        output_size = self.cnn(x).size()
        self.dims = output_size[1]*2
        self.conv_factor= output_size[2] % 5 #should be 1 or 2
        self.fuse_conv_1 = nn.Conv2d(self.dims,self.dims,3)
        self.fuse_conv_2 = nn.Conv2d(self.dims,self.dims,3)
        self.fuse_conv_3 = nn.Conv2d(self.dims,self.dims,2)
        self.fuse_fc = nn.Linear(self.dims*(self.conv_factor**2), 2)
        self.classifier = nn.LogSoftmax(dim=1)

    def forward(self,left_image, right_image):
        batch_size = left_image.size()[0]
        left = self.cnn(left_image)
        right = self.cnn(right_image)
        x = torch.cat((left,right),1)
        x = self.fuse_conv_1(x)
        x = self.fuse_conv_2(x)
        x = self.fuse_conv_3(x)
        x = x.view(batch_size,self.dims*(self.conv_factor**2))
        x = self.fuse_fc(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    net = SsCnn(models.alexnet)
    x = torch.randn([3,224,224]).unsqueeze(0)
    fwd =  net(x,x)
    print(fwd.size())