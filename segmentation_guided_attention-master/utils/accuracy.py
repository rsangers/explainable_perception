import torch.utils.data
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

class RankAccuracy(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._num_correct = None
        self._num_examples = None
        super(RankAccuracy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(RankAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        (rank_left, rank_right, label) = output
        rank_left = rank_left.squeeze()
        rank_right = rank_right.squeeze()
        index_mask = label != 0
        aux_label = label[index_mask]
        diff = rank_left[index_mask] - rank_right[index_mask]
        correct_left = (aux_label == 1) & (diff > 0)
        correct_right = (aux_label == -1) & (diff < 0)
        self._num_correct += torch.sum(correct_left + correct_right).item()
        self._num_examples += aux_label.size()[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._num_correct / self._num_examples

if __name__ == '__main__':
    import torch
    torch.manual_seed(8)

    m = RankAccuracy()
    output = (
        torch.Tensor([1,2,3,1]),
        torch.Tensor([0,3,1,3]),
        torch.Tensor([1,-1,0,1])
    )

    m.update(output)
    m.update(output)
    res = m.compute()

    print(m._num_correct, m._num_examples, res)