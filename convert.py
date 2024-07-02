import json
import pickle
from itertools import product
from pathlib import Path
from typing import Union

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


def standard_graph(
    graph: Union[nx.MultiDiGraph, nx.DiGraph], scores: np.array, score_name: str
) -> nx.DiGraph:
    standard = nx.DiGraph()
    standard.add_nodes_from(graph.nodes)
    nx.set_node_attributes(standard, {n: s for n, s in enumerate(scores)}, score_name)

    if type(graph) is nx.DiGraph:
        edges = [
            (u, v, {"distance": 1, "virtual": False, "weight": 1})
            for u, v in graph.edges
        ]
    elif type(graph) is nx.MultiDiGraph:
        edges = {(u, v): list() for u, v, _ in graph.edges}
        for u, v, k in graph.edges:
            edges[(u, v)].append(k)

        edges = [
            (u, v, {"weight": max(list_k) + 1, "virtual": False, "distance": 1})
            for (u, v), list_k in edges.items()
        ]
    else:
        raise RuntimeError(
            f"Graph should either be a DiGraph or a MultiDiGraph, got {type(graph)}"
        )

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
        graphs, scores_ordering, _, scores = pickle.load(fopen)

    scores = scores.T
    scores_ordering = [np.array(o) for o in scores_ordering]
    scores = [s[np.argsort(o)] for s, o in zip(scores, scores_ordering)]
    return graphs, scores


# gtype = "SF"
# score_name = "betweenness"
# mode = "test"
# graphs, scores = load_data(gtype, score_name, mode)
# ids = unique_graphs(graphs)
# graphs = [standard_graph(graphs[i], scores[i], score_name) for i in ids]
# root_dir = Path(f"./jsons/{gtype}/{score_name}/{mode}")
# save_graphs(graphs, root_dir)

for gtype, score_name, mode in product(
    ["SF", "ER", "GRP"], ["betweenness", "closeness"], ["train", "test"]
):
    graphs, scores = load_data(gtype, score_name, mode)
    ids = unique_graphs(graphs)
    graphs = [standard_graph(graphs[i], scores[i], score_name) for i in ids]

    root_dir = Path(f"./jsons/{gtype}/{score_name}/{mode}")
    save_graphs(graphs, root_dir)
