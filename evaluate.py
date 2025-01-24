from pathlib import Path
import torch
import argparse

from betweenness import load_new_data, test
from model_bet import GNN_Bet
from utils import graph_to_adj_bet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN_Bet(**checkpoint["hyperparams"])
    model.load_state_dict(checkpoint["state-dict"])
    model.to(device)

    (
        list_graph,
        list_n_seq,
        list_num_node,
        bc_mat,
        diameters,
        model_size,
    ) = load_new_data(args.dataset)
    list_adj, list_adj_t = graph_to_adj_bet(
        list_graph,
        list_n_seq,
        list_num_node,
        model_size,
        disable_preprocess=checkpoint["disable-preprocess"],
    )

    metrics = test(model, list_adj, list_adj_t, list_num_node, bc_mat, diameters)
    print("")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.3f}")
