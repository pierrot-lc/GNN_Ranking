import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
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
    print("Loading data...")
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


def load_new_data(data_path):
    graphs = []
    for json_path in data_path.glob("*.json"):
        with open(json_path, "r") as json_file:
            json_graph = json.load(json_file)

        g = nx.node_link_graph(json_graph)
        g = g.to_directed()
        graphs.append(g)

    list_n_sequence = [np.arange(len(g)) for g in graphs]
    list_node_num = [len(g) for g in graphs]
    model_size = max(list_node_num)
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


def random_permutations(n_sequence, node_num, bet, rng):
    """Randomly permute the node sequence. Do not touch the graph as the node sequence
    is used to order the adjacency matrix later on.
    """
    perm = rng.permutation(node_num)
    n_sequence_perm = n_sequence[perm]
    bet_perm = bet[perm]
    bet_perm = np.pad(bet_perm, (0, len(bet) - len(bet_perm)))
    return n_sequence_perm, bet_perm


def train(model, list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train):
    model.train()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    device = next(model.parameters()).device
    model_size = model.ninput
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_t = list_adj_t_train[i]
        true_arr = torch.from_numpy(bc_mat_train[:, i]).float()
        adj = adj.to(device)
        adj_t = adj_t.to(device)
        true_arr = true_arr.to(device)

        optimizer.zero_grad()
        y_out = model(adj, adj_t)
        loss_rank = loss_cal(y_out, true_arr, num_nodes, device, model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()


@torch.no_grad()
def test(model, list_adj, list_adj_t, list_num_node, bc_mat, diameters):
    model.eval()
    loss_val = 0
    list_kt = list()
    list_wkt = list()
    device = next(model.parameters()).device
    model_size = model.ninput
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
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(diameters, list_wkt)
    ax.set_title("Weighted KT-scores")
    ax.set_xlabel("Diameters")
    ax.set_ylabel("Weighted KT-scores")
    fig.tight_layout(pad=2.0)
    metrics["weighted-kt-scores"] = wandb.Image(fig)
    plt.close(fig)

    return metrics


if __name__ == "__main__":
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
    parser.add_argument("--hidden-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--disable-preprocess", action="store_true")
    parser.add_argument("--total-epochs", type=int, default=15)
    parser.add_argument("--mode", default="offline")
    parser.add_argument("--group", default=None)
    parser.add_argument(
        "--random-permutations",
        type=int,
        default=0,
        help="49 for big scale-free graphs, 5 for synthetic graphs",
    )
    args = parser.parse_args()

    (
        list_graph_train,
        list_n_seq_train,
        list_num_node_train,
        bc_mat_train,
        diameters_train,
        model_size_train,
    ) = load_new_data(args.train_dataset)

    (
        list_graph_test,
        list_n_seq_test,
        list_num_node_test,
        bc_mat_test,
        diameters_test,
        model_size_test,
    ) = load_new_data(args.test_dataset)

    # Pad to the biggest of the train/test graphs.
    model_size = max(model_size_train, model_size_test)
    bc_mat_train = np.pad(
        bc_mat_train, ((0, model_size - bc_mat_train.shape[0]), (0, 0))
    )
    bc_mat_test = np.pad(bc_mat_test, ((0, model_size - bc_mat_test.shape[0]), (0, 0)))

    # Generate random permutations of the graph if necessary.
    if args.random_permutations > 0:
        print("Random permutations")
        rng = np.random.default_rng(args.random_permutations)

        for i in range(len(list_graph_train)):
            for _ in range(args.random_permutations):
                n_seq, bet = random_permutations(
                    list_n_seq_train[i], list_num_node_train[i], bc_mat_train[:, i], rng
                )
                list_graph_train.append(list_graph_train[i])
                list_n_seq_train.append(n_seq)
                list_num_node_train.append(list_num_node_train[i])
                bc_mat_train = np.concatenate((bc_mat_train, bet[:, None]), axis=1)
                diameters_train.append(diameters_train[i])

    # Get adjacency matrices from graphs.
    print("Graphs to adjacency conversion")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = GNN_Bet(ninput=model_size, nhid=args.hidden_size, dropout=0.6)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    with wandb.init(
        project="gnn-ranking",
        group=args.group,
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

        for step in tqdm(range(args.total_epochs), desc="Training"):
            train(
                model,
                list_adj_train,
                list_adj_t_train,
                list_num_node_train,
                bc_mat_train,
            )

            # Training stats.
            metrics = {
                "train": test(
                    model,
                    list_adj_train,
                    list_adj_t_train,
                    list_num_node_train,
                    bc_mat_train,
                    diameters_train,
                ),
                "val": test(
                    model,
                    list_adj_test,
                    list_adj_t_test,
                    list_num_node_test,
                    bc_mat_test,
                    diameters_test,
                ),
            }
            logger.log(metrics, step=step)

        logger.summary["train.final-loss"] = metrics["train"]["loss"]
        logger.summary["val.final-loss"] = metrics["val"]["loss"]
        logger.summary["train.final-KT-score"] = metrics["train"]["KT-score"]
        logger.summary["val.final-KT-score"] = metrics["val"]["KT-score"]

        torch.save(
            {
                "hyperparams": {
                    "ninput": model_size,
                    "nhid": args.hidden_size,
                    "dropout": 0.6,
                },
                "disable-preprocess": args.disable_preprocess,
                "state-dict": model.cpu().state_dict(),
            },
            "models/model.pth",
        )
