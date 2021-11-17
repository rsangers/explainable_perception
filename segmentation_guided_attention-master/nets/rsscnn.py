import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np


class RSsCnn(nn.Module):

    def __init__(self,model, finetune=False):
        super(RSsCnn, self).__init__()
        self.cnn = model(pretrained=True).features
        if not finetune:
            for param in self.cnn.parameters():  # freeze cnn params
                param.requires_grad = False
        x = torch.randn([3,244,244]).unsqueeze(0)
        output_size = self.cnn(x).size()
        self.dims = output_size[1]*2
        self.cnn_size = output_size
        self.conv_factor= output_size[2] - 5 #should be 1 or 2
        self.fuse_conv_1 = nn.Conv2d(self.dims,self.dims,3)
        self.fuse_conv_2 = nn.Conv2d(self.dims,self.dims,3)
        self.fuse_conv_3 = nn.Conv2d(self.dims,self.dims,2)
        self.fuse_fc = nn.Linear(self.dims*(self.conv_factor**2), 2)
        self.classifier = nn.LogSoftmax(dim=1)
        self.rank_fc_1 = nn.Linear(self.cnn_size[1]*self.cnn_size[2]*self.cnn_size[3], 4096)
        self.rank_fc_out = nn.Linear(4096, 1)
        self.conv_drop  = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.drop  = nn.Dropout(0.3)

    def forward(self,left_image, right_image):
        batch_size = left_image.size()[0]
        left = self.cnn(left_image)
        right = self.cnn(right_image)
        x = torch.cat((left,right),1)
        x = self.fuse_conv_1(x)
        x = self.conv_drop(x)
        x = self.fuse_conv_2(x)
        x = self.conv_drop(x)
        x = self.fuse_conv_3(x)
        x = self.conv_drop(x)
        x = x.view(batch_size,self.dims*(self.conv_factor**2))
        x_clf = self.fuse_fc(x)
        x_clf = self.classifier(x_clf)
        x_rank_left = left.view(batch_size,self.cnn_size[1]*self.cnn_size[2]*self.cnn_size[3])
        x_rank_right = right.view(batch_size,self.cnn_size[1]*self.cnn_size[2]*self.cnn_size[3])
        x_rank_left = self.rank_fc_1(x_rank_left)
        x_rank_left = self.relu(x_rank_left)
        x_rank_left = self.drop(x_rank_left)
        x_rank_right = self.rank_fc_1(x_rank_right)
        x_rank_right = self.relu(x_rank_right)
        x_rank_right = self.drop(x_rank_right)
        x_rank_left = self.rank_fc_out(x_rank_left)
        x_rank_right = self.rank_fc_out(x_rank_right)
        return x_clf,x_rank_left, x_rank_right

if __name__ == '__main__':
    from torchviz import make_dot
    net = RSsCnn(models.alexnet)
    x = torch.randn([3,244,244]).unsqueeze(0)
    y = torch.randn([3,244,244]).unsqueeze(0)
    fwd =  net(x,y)
    print(fwd)
