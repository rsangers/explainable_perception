import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np


class MyTransformer(nn.Module):

    def __init__(self, model):
        super(MyTransformer, self).__init__()
        self.transformer = model

        x = torch.randn([3, 224, 224]).unsqueeze(0)
        output_size = self.transformer(x).size()
        self.dims = output_size[1] * 2
        self.transformer_size = output_size

        self.transformer.head = nn.Linear(self.transformer.head.in_features, 1)

    def forward(self, left_batch, right_batch=None):
        if right_batch is None:
            return self.transformer(left_batch).unsqueeze(0).unsqueeze(0)
        else:
            return {
                'left': self.single_forward(left_batch),
                'right': self.single_forward(right_batch)
            }

    def single_forward(self, image):
        batch_size = image.size()[0]
        x = self.transformer(image)

        return {
            'output': x
        }


if __name__ == '__main__':
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

    net = MyTransformer(model)
    x = torch.randn([3, 224, 224]).unsqueeze(0)
    y = torch.randn([3, 224, 224]).unsqueeze(0)
    fwd = net(x, y)
    print(fwd)