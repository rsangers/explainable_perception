import numpy as np
from sklearn.metrics import label_ranking_average_precision_score as rank_score
import torch

def compute_ranking_loss(x_left,x_right,label,loss):
    output_left = x_left.view(x_left.size()[0])
    output_right = x_right.view(x_right.size()[0])
    return loss(output_left, output_right, label)

def compute_multiple_ranking_loss(x_left, x_right, label, loss, attr_ids):
    attr_tensor = attr_ids.unsqueeze(1)
    output_left = torch.gather(x_left, 1, attr_tensor).squeeze()
    output_right = torch.gather(x_right, 1, attr_tensor).squeeze()
    return loss(output_left, output_right, label)

def compute_ranking_accuracy(x_left, x_right, label):
    indexes = label!=0
    aux_label = label[indexes]
    if aux_label.size()[0] == 0: return 0.5 # extremely rare only ties case
    indexes = indexes.cpu().detach().numpy()
    indexes.dtype=bool
    rank_pairs = np.array(list(zip(x_left,x_right)))[indexes]
    label_matrix = aux_label.clone().cpu().detach().numpy()
    dup = np.zeros(label_matrix.shape)
    label_matrix[label_matrix==-1] = 0
    dup[label_matrix==0] = 1
    label_matrix = np.hstack((np.array([label_matrix]).T,np.array([dup]).T))
    return (rank_score(label_matrix,rank_pairs) - 0.5)/0.5

def compute_multiple_ranking_accuracy(x_left, x_right, label, attr_ids):
    attr_tensor = attr_ids.unsqueeze(1)
    output_left = torch.gather(x_left, 1, attr_tensor)
    output_right = torch.gather(x_right, 1, attr_tensor)
    return compute_ranking_accuracy(output_left,output_right, label)