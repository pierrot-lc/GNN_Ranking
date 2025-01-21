import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GNN_Layer(Module):
    """
    Layer defined for GNN-Bet
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GNN_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GNN_Layer_Init(Module):
    """
    First layer of GNN_Init, for embedding lookup
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GNN_Layer_Init, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # NOTE: This `weight` parameter is dumb. They learn a different embedding for
        # each node based on their position in the adjacency matrix. The resulting
        # operation is not permutation equivariant and limited to a predefined maximum
        # number of nodes.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj):
        support = self.weight
        # support = torch.ones((self.in_features, self.out_features), device=adj.device)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MLP(Module):
    def __init__(self, nhid, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.linear1 = torch.nn.Linear(nhid, 2 * nhid)
        self.linear2 = torch.nn.Linear(2 * nhid, 2 * nhid)
        self.linear3 = torch.nn.Linear(2 * nhid, 1)

    def forward(self, input_vec, dropout):
        # NOTE: The original implementation called F.dropout without setting the train
        # argument to True (False by default). When the argument is set to False, this
        # function does nothing (as of v0.4.1 of PyTorch). This default argument changed
        # to True in newer versions so I just removed the call to F.dropout (the old
        # version is too hard to install).

        # See the original (v0.4.1) F.dropout here:
        # https://github.com/pytorch/pytorch/blob/v0.4.1/torch/nn/functional.py#L594
        # https://github.com/pytorch/pytorch/blob/v0.4.1/torch/nn/_functions/dropout.py#L27

        score_temp = F.relu(self.linear1(input_vec))
        # score_temp = F.dropout(score_temp, self.dropout)
        score_temp = F.relu(self.linear2(score_temp))
        # score_temp = F.dropout(score_temp, self.dropout)
        score_temp = self.linear3(score_temp)

        return score_temp
