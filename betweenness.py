import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model_bet import GNN_Bet

torch.manual_seed(20)
import argparse

# Loading graph data
parser = argparse.ArgumentParser()
parser.add_argument("--g", default="SF")
args = parser.parse_args()
gtype = args.g
print(gtype)
if gtype == "SF":
    data_path = "./datasets/data_splits/SF/betweenness/"
    print("Scale-free graphs selected.")

elif gtype == "ER":
    data_path = "./datasets/data_splits/ER/betweenness/"
    print("Erdos-Renyi random graphs selected.")
elif gtype == "GRP":
    data_path = "./datasets/data_splits/GRP/betweenness/"
    print("Gaussian Random Partition graphs selected.")


# Load training data
print(f"Loading data...")
with open(data_path + "training.pickle", "rb") as fopen:
    list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train = pickle.load(
        fopen
    )


with open(data_path + "test.pickle", "rb") as fopen:
    list_graph_test, list_n_seq_test, list_num_node_test, bc_mat_test = pickle.load(
        fopen
    )

model_size = 1_000_000
# Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.")

list_adj_train, list_adj_t_train = graph_to_adj_bet(
    list_graph_train, list_n_seq_train, list_num_node_train, model_size
)
list_adj_test, list_adj_t_test = graph_to_adj_bet(
    list_graph_test, list_n_seq_test, list_num_node_test, model_size
)


def train(list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train):
    model.train()
    total_count_train = list()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_t = list_adj_t_train[i]
        adj = adj.to(device)
        adj_t = adj_t.to(device)

        optimizer.zero_grad()

        y_out = model(adj, adj_t)
        true_arr = torch.from_numpy(bc_mat_train[:, i]).float()
        true_val = true_arr.to(device)

        loss_rank = loss_cal(y_out, true_val, num_nodes, device, model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()


def test(list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test):
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj = adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]

        y_out = model(adj, adj_t)

        true_arr = torch.from_numpy(bc_mat_test[:, j]).float()
        true_val = true_arr.to(device)

        kt = ranking_correlation(y_out, true_val, num_nodes, model_size)
        list_kt.append(kt)
        # g_tmp = list_graph_test[j]
        # print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")

    print(
        f"   Average KT score on test graphs is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}"
    )


# Model parameters
hidden = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN_Bet(ninput=model_size, nhid=hidden, dropout=0.6)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
num_epoch = 15

print("Training")
print(f"Total Number of epoches: {num_epoch}")
for e in range(num_epoch):
    print(f"Epoch number: {e + 1}/{num_epoch}")
    train(list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train)

    # to check test loss while training
    with torch.no_grad():
        test(list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test)
# test on 10 test graphs and print average KT Score and its stanard deviation
# with torch.no_grad():
#    test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)


from pathlib import Path

Path("models").mkdir(exist_ok=True)
torch.save(
    {
        "hyperparams": {
            "ninput": model_size,
            "nhid": hidden,
            "dropout": 0.6,
        },
        "disable-preprocess": False,
        "state-dict": model.cpu().state_dict(),
    },
    "models/model-repro.pth",
)

