import json
import pickle
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np


def unique_graphs(list_graphs: list) -> list:
    found = []
    ids = []

    for id, graph in enumerate(list_graphs):
        if not any(nx.utils.graphs_equal(graph, g) for g in found):
            found.append(graph)
            ids.append(id)

    return ids


def standard_graph(adj: np.ndarray, scores: np.array, score_name: str) -> nx.DiGraph:
    standard = nx.DiGraph()
    standard.add_nodes_from(node_id for node_id in range(len(scores)))
    nx.set_node_attributes(
        standard, {n: float(s) for n, s in enumerate(scores)}, score_name
    )

    edges = [
        (int(v), int(u), {"distance": 1, "virtual": False, "weight": int(adj[u, v])})
        for u, v in zip(*np.nonzero(adj))
    ]
    standard.add_edges_from(edges)

    return standard


def save_graphs(graphs: list, root_dir: Path):
    root_dir.mkdir(parents=True, exist_ok=False)

    for graph_id, graph in enumerate(graphs):
        path = root_dir / f"{graph_id}.json"
        data = nx.node_link_data(graph)
        json_data = json.dumps(data, indent=2)
        with open(path, "w") as json_file:
            json_file.write(json_data)


def load_data(gtype: str, score_name: str, mode: str):
    if mode == "train":
        mode = "training"

    data_path = f"./datasets/data_splits/{gtype}/{score_name}/{mode}.pickle"
    with open(data_path, "rb") as fopen:
        graphs, node_sequences, _, scores = pickle.load(fopen)

    adjs = [nx.adjacency_matrix(g, nodelist=n) for g, n in zip(graphs, node_sequences)]
    return graphs, adjs, scores.T


# gtype = "SF"
# score_name = "betweenness"
# mode = "test"
# graphs, adjs, scores = load_data(gtype, score_name, mode)
# ids = unique_graphs(graphs)
# graphs = [standard_graph(adjs[i], scores[i], score_name) for i in ids]
# root_dir = Path(f"./jsons/{gtype}/{score_name}/{mode}")
# save_graphs(graphs, root_dir)

for gtype, score_name, mode in product(
    ["SF", "ER", "GRP"], ["betweenness", "closeness"], ["train", "test"]
):
    print(gtype, score_name, mode)
    graphs, adjs, scores = load_data(gtype, score_name, mode)
    ids = unique_graphs(graphs)
    graphs = [standard_graph(adjs[i], scores[i], score_name) for i in ids]

    root_dir = Path(f"./jsons/{gtype}/{score_name}/{mode}")
    save_graphs(graphs, root_dir)
