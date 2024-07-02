import argparse
import json
import pickle
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from model_bet import GNN_Bet
from utils import *

torch.manual_seed(20)


def load_original_data():
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
        list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train = (
            pickle.load(fopen)
        )

    with open(data_path + "test.pickle", "rb") as fopen:
        list_graph_test, list_n_seq_test, list_num_node_test, bc_mat_test = pickle.load(
            fopen
        )

    model_size = 10000

    return (
        list_graph_train,
        list_graph_test,
        list_n_seq_train,
        list_n_seq_test,
        list_num_node_train,
        list_num_node_test,
        bc_mat_train,
        bc_mat_test,
        model_size,
    )


def remove_virtual_edges(graphs: list) -> list:
    for graph in graphs:
        to_remove = []
        for u, v, d in graph.edges(data=True):
            if d["virtual"]:
                to_remove.append((u, v))

        for u, v in to_remove:
            graph.remove_edge(u, v)

    return graphs


def load_new_data(data_path, remove_virtual: bool = True):
    graphs = []
    for json_path in data_path.glob("*.json"):
        with open(json_path, "r") as json_file:
            json_graph = json.load(json_file)

        graphs.append(nx.node_link_graph(json_graph))

    if remove_virtual:
        graphs = remove_virtual_edges(graphs)

    list_n_sequence = [np.arange(len(g)) for g in graphs]
    list_node_num = [len(g) for g in graphs]
    model_size = 10000
    cent_mat = [np.array([g.nodes[n]["betweenness"] for n in g.nodes]) for g in graphs]
    cent_mat = np.stack(
        [
            np.pad(
                m,
                (0, model_size - len(m)),
            )
            for m in cent_mat
        ]
    )
    cent_mat = cent_mat.transpose()

    return (
        graphs,
        list_n_sequence,
        list_node_num,
        cent_mat,
        model_size,
    )


(
    list_graph_train,
    list_graph_test,
    list_n_seq_train,
    list_n_seq_test,
    list_num_node_train,
    list_num_node_test,
    bc_mat_train,
    bc_mat_test,
    model_size,
) = load_original_data()


# data_path = Path("./jsons/ER/betweenness/")
# data_path = Path("../gnn-ranking/datasets/geometric_1000-0.4/")
# (
#     list_graph_train,
#     list_n_seq_train,
#     list_num_node_train,
#     bc_mat_train,
#     model_size,
# ) = load_new_data(data_path / "train")
#
# (
#     list_graph_test,
#     list_n_seq_test,
#     list_num_node_test,
#     bc_mat_test,
#     model_size,
# ) = load_new_data(data_path / "test")

# Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.")

list_adj_train, list_adj_t_train = graph_to_adj_bet(
    list_graph_train,
    list_n_seq_train,
    list_num_node_train,
    model_size,
    disable_preprocess=True,
)
list_adj_test, list_adj_t_test = graph_to_adj_bet(
    list_graph_test,
    list_n_seq_test,
    list_num_node_test,
    model_size,
    disable_preprocess=True,
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

        # assert torch.all(adj_t.to_dense().T == adj.to_dense())

        optimizer.zero_grad()

        y_out = model(adj, adj_t)
        true_arr = torch.from_numpy(bc_mat_train[:, i]).float()
        true_val = true_arr.to(device)

        loss_rank = loss_cal(y_out, true_val, num_nodes, device, model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()

    print("loss train:", loss_train / num_samples_train)


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
device = "cpu"
model = GNN_Bet(ninput=model_size, nhid=hidden, dropout=0.0)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
num_epoch = 100

tot_params = [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
tot_params = np.sum(tot_params)
print(f"Total params: {tot_params:,}")

# list_adj_train = list_adj_train[:1]
# list_adj_t_train = list_adj_t_train[:1]
# list_num_node_train = list_num_node_train[:1]
# bc_mat_train = bc_mat_train[:, :1]
#
# list_adj_test = list_adj_train
# list_adj_t_test = list_adj_t_train
# list_num_node_test = list_num_node_train
# bc_mat_test = bc_mat_train

print(f"Training on {device}")
print(f"Total Number of epoches: {num_epoch}")
print(f"Total training examples: {len(list_adj_train)}")
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    train(list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train)

    # to check test loss while training
    with torch.no_grad():
        test(list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test)
# test on 10 test graphs and print average KT Score and its stanard deviation
# with torch.no_grad():
#    test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)
