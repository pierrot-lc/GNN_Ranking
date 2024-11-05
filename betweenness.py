import argparse
import matplotlib.pyplot as plt
import json
import pickle
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
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
    cent_mat = np.stack([np.pad(m, (0, model_size - len(m))) for m in cent_mat])
    cent_mat = cent_mat.transpose()

    diameters = []
    for graph in graphs:
        largest_component = max(nx.weakly_connected_components(graph), key=len)
        largest_component = graph.subgraph(largest_component).copy()
        largest_component = nx.to_undirected(largest_component)
        diameter = nx.algorithms.approximation.diameter(largest_component, seed=0)
        diameters.append(diameter)

    return (
        graphs,
        list_n_sequence,
        list_node_num,
        cent_mat,
        diameters,
        model_size,
    )


# (
#     list_graph_train,
#     list_graph_test,
#     list_n_seq_train,
#     list_n_seq_test,
#     list_num_node_train,
#     list_num_node_test,
#     bc_mat_train,
#     bc_mat_test,
#     model_size,
# ) = load_original_data()

parser = argparse.ArgumentParser()
parser.add_argument("--train-dataset", type=Path, required=True)
parser.add_argument("--test-dataset", type=Path, required=True)
parser.add_argument("--hidden-size", type=int, default=24)
parser.add_argument("--disable-preprocess", action="store_true")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--total-iters", type=int, default=100_000)
parser.add_argument("--eval-freq", type=int, default=1000)
parser.add_argument("--eval-iters", type=int, default=100)
parser.add_argument("--mode", default="offline")
args = parser.parse_args()

(
    list_graph_train,
    list_n_seq_train,
    list_num_node_train,
    bc_mat_train,
    diameters_train,
    model_size,
) = load_new_data(args.train_dataset)

(
    list_graph_test,
    list_n_seq_test,
    list_num_node_test,
    bc_mat_test,
    diameters_test,
    model_size,
) = load_new_data(args.test_dataset)

# Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.")

list_adj_train, list_adj_t_train = graph_to_adj_bet(
    list_graph_train,
    list_n_seq_train,
    list_num_node_train,
    model_size,
    disable_preprocess=args.disable_preprocess,
)
list_adj_test, list_adj_t_test = graph_to_adj_bet(
    list_graph_test,
    list_n_seq_test,
    list_num_node_test,
    model_size,
    disable_preprocess=args.disable_preprocess,
)


def train(list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train):
    model.train()
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


@torch.no_grad()
def test(list_adj, list_adj_t, list_num_node, bc_mat, diameters):
    model.eval()
    loss_val = 0
    list_kt = list()
    list_wkt = list()
    num_samples_test = len(list_adj)
    for j in range(num_samples_test):
        adj = list_adj[j]
        adj_t = list_adj_t[j]
        adj = adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node[j]

        y_out = model(adj, adj_t)

        true_arr = torch.from_numpy(bc_mat[:, j]).float()
        true_val = true_arr.to(device)

        loss_rank = loss_cal(y_out, true_val, num_nodes, device, model_size)
        loss_val = loss_val + float(loss_rank)

        kt, wkt = ranking_correlation(y_out, true_val, num_nodes, model_size)
        list_kt.append(kt)
        list_wkt.append(wkt)

    metrics = {
        "loss": loss_val / num_samples_test,
        "KT-score": np.mean(np.array(list_kt)),
        "Weighted KT-score": np.mean(np.array(list_wkt)),
    }

    fig, ax = plt.subplots()
    ax.scatter(diameters, list_kt)
    ax.set_title("KT-scores")
    ax.set_xlabel("Diameters")
    ax.set_ylabel("KT-scores")
    fig.tight_layout(pad=2.0)
    metrics["kt-scores"] = wandb.Image(fig)

    fig, ax = plt.subplots()
    ax.scatter(diameters, list_wkt)
    ax.set_title("Weighted KT-scores")
    ax.set_xlabel("Diameters")
    ax.set_ylabel("Weighted KT-scores")
    fig.tight_layout(pad=2.0)
    metrics["weighted-kt-scores"] = wandb.Image(fig)

    return metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model = GNN_Bet(ninput=model_size, nhid=args.hidden_size, dropout=0.0)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

with wandb.init(
    project="gnn-ranking",
    group="original-implementation",
    config=vars(args),
    entity="neuralcombopt",
    mode=args.mode,
) as logger:
    tot_params = [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    tot_params = np.sum(tot_params)

    logger.summary["params"] = tot_params

    print(f"Training on {device}")
    print(f"Total params: {tot_params:,}")
    print(f"Total training examples: {len(list_adj_train):,}")

    logger.define_metric("train.loss", summary="min")
    logger.define_metric("val.loss", summary="min")

    logger.define_metric("train.KT-score", summary="max")
    logger.define_metric("val.KT-score", summary="max")

    logger.define_metric("train.Weighted KT-score", summary="max")
    logger.define_metric("val.Weighted KT-score", summary="max")

    for step in tqdm(range(args.total_iters), desc="Training"):
        optimizer.zero_grad()

        # Cumulate the gradients for a batch of samples.
        for _ in range(args.batch_size):
            # Train the model on one sample.
            i = random.randint(0, len(list_adj_t_train) - 1)
            adj = list_adj_train[i]
            num_nodes = list_num_node_train[i]
            adj_t = list_adj_t_train[i]

            adj = adj.to(device)
            adj_t = adj_t.to(device)

            y_out = model(adj, adj_t)
            true_arr = torch.from_numpy(bc_mat_train[:, i]).float()
            true_val = true_arr.to(device)

            loss_rank = loss_cal(y_out, true_val, num_nodes, device, model_size)
            loss_rank.backward()

        optimizer.step()

        if step % args.eval_freq == 0:
            ids = random.choices(range(len(list_adj_train)), k=args.eval_iters)
            list_adj = [list_adj_train[i] for i in ids]
            list_adj_t = [list_adj_t_train[i] for i in ids]
            list_num_node = [list_num_node_train[i] for i in ids]
            bc_mat = np.stack([bc_mat_train[:, i] for i in ids], axis=1)
            diameters = [diameters_train[i] for i in ids]
            metrics = test(list_adj, list_adj_t, list_num_node, bc_mat, diameters)
            logger.log({"train": metrics}, step=step)

            ids = random.choices(range(len(list_adj_test)), k=args.eval_iters)
            list_adj = [list_adj_test[i] for i in ids]
            list_adj_t = [list_adj_t_test[i] for i in ids]
            list_num_node = [list_num_node_test[i] for i in ids]
            bc_mat = np.stack([bc_mat_test[:, i] for i in ids], axis=1)
            diameters = [diameters_test[i] for i in ids]
            metrics = test(list_adj, list_adj_t, list_num_node, bc_mat, diameters)
            logger.log({"val": metrics}, step=step)
