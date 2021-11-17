import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

class AttentionRCNN(nn.Module):

    def __init__(self,model, finetune=False, n_layers=1, n_heads=1):
        super(AttentionRCNN, self).__init__()
        try:
            self.cnn = model(pretrained=True).features
        except AttributeError:
            self.cnn = nn.Sequential(*list(model(pretrained=True).children())[:-2])
        if not finetune:
            for param in self.cnn.parameters():  # freeze cnn params
                param.requires_grad = False
        self.n_layers = n_layers
        self.n_heads = n_heads
        x = torch.randn([3,244,244]).unsqueeze(0)
        output_size = self.cnn(x).size()
        self.dims = output_size[1]*2
        self.cnn_size = output_size
        self.attentions = nn.ModuleList([nn.MultiheadAttention(self.cnn_size[1], self.n_heads, dropout=0.3,) for _ in range(self.n_layers)])
        
        self.rank_fc = nn.Linear(self.cnn_size[2]*self.cnn_size[3]*self.cnn_size[1], 500)
        self.relu = nn.ReLU()
        self.drop  = nn.Dropout(0.3)
        self.rank_fc_out = nn.Linear(500, 1)

    def forward(self,left_batch, right_batch):
        return {
            'left': self.single_forward(left_batch),
            'right': self.single_forward(right_batch)
        }

    def single_forward(self,image):
        batch_size = image.size()[0]
        x = self.cnn(image)
        x = self.drop(x)
        x = x.permute([2,3,0,1])
        x = x.view(self.cnn_size[2]*self.cnn_size[3],batch_size,self.cnn_size[1])
        attn_list = []
        for attention in self.attentions:
            x, attn_weights = attention(x, x, x)
            attn_list.append(attn_weights)
        x = x.permute([1,0,2]).contiguous().view(batch_size,self.cnn_size[2]*self.cnn_size[3]*self.cnn_size[1])
        x = self.drop(self.relu(self.rank_fc(x)))
        x = self.rank_fc_out(x)
        return {
            'output': x,
            'attention': attn_list
        }

if __name__ == '__main__':
    net = AttentionRCNN(models.resnet50)
    x = torch.randn([3,244,244]).unsqueeze(0)
    y = torch.randn([3,244,244]).unsqueeze(0)
    fwd =  net(x,y)
    print(fwd)