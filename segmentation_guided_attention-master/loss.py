import torch
from torch import nn

class RankingLoss(nn.Module):

    def __init__(self, margin=1, tie_margin=1):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.tie_margin = tie_margin

    def forward(self,x_left,x_right,y):
        diff = x_left - x_right
        bin_y = torch.abs(y)
        tie_y =  1 - bin_y
        tie_loss = torch.clamp((torch.abs(diff) - self.tie_margin)*tie_y , min=0)
        rank_loss = torch.clamp(-y*(diff) + bin_y*self.margin, min=0)
        return torch.mean(rank_loss + tie_loss)



class LogSumExpLoss(nn.Module):

    def forward(self,x_left,x_right,y):
        diff = x_left - x_right
        bin_y = torch.abs(y)
        tie_y =  1 - bin_y
        rank_loss = torch.log(1 + torch.exp(-y*diff)) * bin_y
        tie_loss = torch.log(torch.exp(diff) +  torch.exp(-diff)) * tie_y
        return torch.mean(rank_loss + tie_loss)

class RegressionRegularizer(nn.Module):

    def forward(self, x):
        return torch.mean(1 / x**2)

class DoubleRegressionRegularizer(nn.Module):

    def __init__(self):
        super(DoubleRegressionRegularizer, self).__init__()
        self.reg = RegressionRegularizer()


    def forward(self, x_left, x_right):
        return self.reg(x_left) + self.reg(x_right)

class VarianceRegularizer(nn.Module):
    def forward(self, x_left, x_right):
        outputs = torch.cat((x_left,x_right))
        variance = torch.mean((outputs - torch.mean(outputs)) ** 2)
        return 1/variance

class RegularizedLoss(nn.Module):

    def __init__(self, loss, regularizer, alpha=0.1):
        super(RegularizedLoss, self).__init__()
        self.loss = loss
        self.regularizer = regularizer
        self.alpha = alpha

    def forward(self, x_left, x_right, y):
        return self.loss(x_left, x_right, y) + self.alpha * (self.regularizer(x_left, x_right))


if __name__ == '__main__':
    ls = LogSumExpLoss()
    x_left = torch.Tensor([3])
    x_right = torch.Tensor([1.2])
    y = torch.Tensor([0])
    print(ls(x_left,x_right,y))
    reg = VarianceRegularizer()
    print(reg(x_left,x_right))
    final_loss = RegularizedLoss(ls,reg)
    print(final_loss(x_left,x_right,y))

