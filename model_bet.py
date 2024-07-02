import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from layer import GNN_Layer
from layer import GNN_Layer_Init
from layer import MLP
import torch


class GNN_Bet(nn.Module):
    def __init__(self, ninput, nhid, dropout):
        super(GNN_Bet, self).__init__()

        self.gc1 = GNN_Layer_Init(ninput, nhid)
        self.gc2 = GNN_Layer(nhid, nhid)
        self.gc3 = GNN_Layer(nhid, nhid)
        self.gc4 = GNN_Layer(nhid, nhid)
        self.gc5 = GNN_Layer(nhid, nhid)
        # self.gc6 = GNN_Layer(nhid,nhid)

        self.dropout = dropout

        self.score_layer = MLP(nhid, self.dropout)

    def forward(self, adj1, adj2):
        # Layers for aggregation operation
        x_1 = F.normalize(F.relu(self.gc1(adj1)), p=2, dim=1)
        x2_1 = F.normalize(F.relu(self.gc1(adj2)), p=2, dim=1)

        x_2 = F.normalize(F.relu(self.gc2(x_1, adj1)), p=2, dim=1)
        x2_2 = F.normalize(F.relu(self.gc2(x2_1, adj2)), p=2, dim=1)

        x_3 = F.normalize(F.relu(self.gc3(x_2, adj1)), p=2, dim=1)
        x2_3 = F.normalize(F.relu(self.gc3(x2_2, adj2)), p=2, dim=1)

        x_4 = F.normalize(F.relu(self.gc4(x_3, adj1)), p=2, dim=1)
        x2_4 = F.normalize(F.relu(self.gc4(x2_3, adj2)), p=2, dim=1)

        x_5 = F.relu(self.gc5(x_4, adj1))
        x2_5 = F.relu(self.gc5(x2_4, adj2))

        # x_5 = self.gc5(x_1, adj1)
        # x2_5 = self.gc5(x2_1, adj2)
        # score_1 = self.score_layer(x_5, self.dropout)
        # score_2 = self.score_layer(x2_5, self.dropout)
        # return score_1 * score_2

        # x_5 = F.normalize(x_5)
        # x2_5 = F.normalize(x2_5)

        # adj1_ = adj1.to_dense().cpu().numpy()
        # np.save("adj", adj1_)
        # np.save("x1", x_1.cpu().numpy())
        # np.save("linear1", self.gc2.weight.data.cpu().numpy())
        # o = torch.mm(x_1, self.gc2.weight)
        # o = torch.spmm(adj1, o)
        # np.save("conv_output", o.detach().cpu().numpy())
        # np.save("conv_output_sav", self.gc2(x_1, adj1).detach().cpu().numpy())
        # np.save("x2", x_2.detach().cpu().numpy())
        # print(x_1[:10])
        # print(x_2[:10])
        # print(x_3[:10])
        # print(x_4[:10])
        # print(torch.std(x_4, dim=0))
        # assert False

        # Score Calculations

        score1_1 = self.score_layer(x_1, self.dropout)
        score1_2 = self.score_layer(x_2, self.dropout)
        score1_3 = self.score_layer(x_3, self.dropout)
        score1_4 = self.score_layer(x_4, self.dropout)
        score1_5 = self.score_layer(x_5, self.dropout)

        score2_1 = self.score_layer(x2_1, self.dropout)
        score2_2 = self.score_layer(x2_2, self.dropout)
        score2_3 = self.score_layer(x2_3, self.dropout)
        score2_4 = self.score_layer(x2_4, self.dropout)
        score2_5 = self.score_layer(x2_5, self.dropout)

        score1 = score1_1 + score1_2 + score1_3 + score1_4 + score1_5
        score2 = score2_1 + score2_2 + score2_3 + score2_4 + score2_5

        x = torch.mul(score1, score2)

        return x
