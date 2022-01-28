import torchvision.models as models
import torch.nn as nn
import torch


class MyCnn(nn.Module):

    def __init__(self, model, finetune=False):
        super(MyCnn, self).__init__()
        try:
            self.cnn = model(pretrained=True).features
        except AttributeError:
            self.cnn = nn.Sequential(*list(model(pretrained=True).children())[:-1])
        if not finetune:
            for param in self.cnn.parameters():  # freeze cnn params
                param.requires_grad = False
        x = torch.randn([3, 244, 244]).unsqueeze(0)
        output_size = self.cnn(x).size()
        self.dims = output_size[1] * 2
        self.cnn_size = output_size
        self.rank_fc_1 = nn.Linear(self.cnn_size[1] * self.cnn_size[2] * self.cnn_size[3], 4096)
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
        batch_size = batch.size()[0]
        x = self.cnn(batch)

        x = x.reshape(batch_size, self.cnn_size[1] * self.cnn_size[2] * self.cnn_size[3])
        x = self.rank_fc_1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.rank_fc_out(x)

        return {
            'output': x
        }


if __name__ == '__main__':
    net = MyCnn(models.resnet50)
    x = torch.randn([3, 244, 244]).unsqueeze(0)
    y = torch.randn([3, 244, 244]).unsqueeze(0)
    fwd = net(x, y)
    print(fwd)
