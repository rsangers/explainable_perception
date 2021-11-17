import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

class RCnn(nn.Module):

    def __init__(self,model, finetune=False):
        super(RCnn, self).__init__()
        try:
            self.cnn = model(pretrained=True).features
        except AttributeError:
            self.cnn = nn.Sequential(*list(model(pretrained=True).children())[:-1])
        if not finetune:
            for param in self.cnn.parameters():  # freeze cnn params
                param.requires_grad = False
        x = torch.randn([3,244,244]).unsqueeze(0)
        output_size = self.cnn(x).size()
        self.dims = output_size[1]*2
        self.cnn_size = output_size
        self.rank_fc_1 = nn.Linear(self.cnn_size[1]*self.cnn_size[2]*self.cnn_size[3], 4096)
        self.rank_fc_out = nn.Linear(4096, 1)
        self.relu = nn.ReLU()
        self.drop  = nn.Dropout(0.3)
        print("Layer shape: ", self.rank_fc_1.weight.shape)

    def forward(self,left_image, right_image=None):
        if right_image is None:
            batch_size = left_image.size()[0]
            left = self.cnn(left_image)
            x_rank_left = left.view(batch_size, self.cnn_size[1] * self.cnn_size[2] * self.cnn_size[3])
            x_rank_left = self.rank_fc_1(x_rank_left)
            x_rank_left = self.relu(x_rank_left)
            x_rank_left = self.drop(x_rank_left)
            x_rank_left = self.rank_fc_out(x_rank_left)
            return x_rank_left.unsqueeze(0).unsqueeze(0)
        else:
            batch_size = left_image.size()[0]
            left = self.cnn(left_image)
            right = self.cnn(right_image)
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
            return {
                'left': { 'output': x_rank_left},
                'right': { 'output': x_rank_right}
            }

if __name__ == '__main__':
    from torchviz import make_dot
    net = RCnn(models.resnet50)
    x = torch.randn([3,244,244]).unsqueeze(0)
    y = torch.randn([3,244,244]).unsqueeze(0)
    fwd =  net(x,y)
    print(fwd)
