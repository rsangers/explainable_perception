import torch.nn as nn
import torch


class MyTransformer(nn.Module):

    def __init__(self, model):
        super(MyTransformer, self).__init__()
        self.transformer = model

        self.transformer.head = nn.Linear(self.transformer.head.in_features, 1)
        self.rank_fc_1 = nn.Linear(self.transformer.head.out_features, 4096)
        self.rank_fc_out = nn.Linear(4096, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, left_batch, right_batch=None):
        if right_batch is None:
            return self.single_forward(left_batch)['output'].unsqueeze(0).unsqueeze(0)
        else:
            return {
                'left': self.single_forward(left_batch),
                'right': self.single_forward(right_batch)
            }

    def single_forward(self, batch):
        x = self.transformer(batch)

        if isinstance(x, tuple):
            x = x[0]

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