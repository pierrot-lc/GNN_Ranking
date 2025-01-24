import argparse
from pathlib import Path

import numpy as np
import torch

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

    model_size = checkpoint["hyperparams"]["ninput"]

    (
        list_graph,
        list_n_seq,
        list_num_node,
        bc_mat,
        diameters,
        _,
    ) = load_new_data(args.dataset)

    # Fit to the model's model_size.
    bc_mat = np.pad(bc_mat, ((0, model_size - len(bc_mat)), (0, 0)))

    list_adj, list_adj_t = graph_to_adj_bet(
        list_graph,
        list_n_seq,
        list_num_node,
        model_size,
        disable_preprocess=checkpoint["disable-preprocess"],
    )

    metrics = test(model, list_adj, list_adj_t, list_num_node, bc_mat, diameters)
    print("\nMetrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.3f}")
