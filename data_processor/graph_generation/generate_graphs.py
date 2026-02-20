import random
import numbers
from collections.abc import Callable
import networkx as nx
import numpy as np
import os


def build_graph(
    n: int,
    degree_mode: str = "equal",                 # 'equal' | 'unequal' | 'custom'
    num_edges: int | None = None,              # fix |E| exactly
    k: int | None = None,                      # used when degree_mode == 'equal'
    m: int = 2,                                # used when degree_mode == 'unequal'
    deg_seq: list[int] | None = None,          # used when degree_mode == 'custom'
    directed: bool = False,
    weighted: bool = False,
    weight_sampler: Callable[[], numbers.Real] | None = None,
    seed: int | None = None
) -> nx.Graph | nx.DiGraph:
    """
    Generic graph factory.

    Parameters
    ----------
    n, directed, weighted : see docstring above
    degree_mode : 'equal' | 'unequal' | 'custom'
        'equal'   – k-regular (undirected) or constant-k-out (directed)
        'unequal' – power-law / scale-free (Barabási–Albert or scale-free digraph)
        'custom'  – `deg_seq` is used to drive nx.configuration_model
    k : int
        Degree to enforce when degree_mode='equal'
    m : int
        Edges to attach for each new node in BA model
    deg_seq : list[int]
        Desired (undirected) degree sequence **or**
        when directed: list[tuple[int,int]] where each item is (out_deg, in_deg)
    weight_sampler : Callable that returns a weight (defaults to U(1, 10))

    Returns
    -------
    G : networkx.Graph or networkx.DiGraph
    """
    rng = np.random.default_rng(seed)
    if weight_sampler is None:
        weight_sampler = lambda: rng.integers(1, 11)          # U{1..10}

        # ---------------- sanity-check the requested size -----------------
    max_e = n * (n - 1) if directed else n * (n - 1) // 2
    if num_edges is not None and not (0 <= num_edges <= max_e):
        raise ValueError(f"num_edges must be in [0, {max_e}] for n = {n}")

    # ------------------------------------------------------------------ build G
    # ❶ Fastest path: explicit |E| → use g(n,m) random graph
    if num_edges is not None and degree_mode != "equal":
        G = nx.gnm_random_graph(n, num_edges, directed=directed, seed=seed)

    # ------------------------------------------------------------------ build G
    if degree_mode == "equal":
        if k is None:
            if num_edges is None:
                k = 2                                          # default fallback
            else:
                # solve k from |E| = (n·k)/2   or   |E| = n·k   (directed)
                k_raw = (num_edges * (2 if not directed else 1)) / n
                k = int(k_raw)
                if not k_raw.is_integer():
                    raise ValueError(
                        "num_edges does not correspond to an integer k "
                        f"(got k = {k_raw:.2f})."
                    )                                          # sensible fallback
        if directed:
            # every node gets *out-degree* = k    (in-degree is variable)
            G = nx.random_k_out_graph(
                n=n, k=k, alpha=0.75,      # α controls “greediness” of hubs
                self_loops=False, seed=seed
            )
            G = nx.DiGraph(G)
        else:
            if k >= n or (k * n) % 2 == 1:
                raise ValueError("Need 0<k<n and k·n even for k-regular graph")
            G = nx.random_regular_graph(d=k, n=n, seed=seed)
            G = nx.Graph(G)

    elif degree_mode == "unequal":
        if directed:
            G = nx.scale_free_graph(n=n, seed=seed)           # MultiDiGraph
            G = nx.DiGraph(G)                                 # collapse multiedges
        else:
            G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
            G = nx.Graph(G)

    elif degree_mode == "custom":
        if deg_seq is None:
            raise ValueError("Provide deg_seq when degree_mode='custom'")

        if directed:
            # deg_seq = [(out1, in1), (out2, in2), ...]
            out_seq, in_seq = zip(*deg_seq)
            multigraph = nx.directed_configuration_model(
                in_seq, out_seq, create_using=nx.MultiDiGraph, seed=seed
            )
            G = nx.DiGraph(multigraph)        # remove multiedges + self-loops
            G.remove_edges_from(nx.selfloop_edges(G))
        else:
            multigraph = nx.configuration_model(
                deg_seq, create_using=nx.MultiGraph, seed=seed
            )
            G = nx.Graph(multigraph)
            G.remove_edges_from(nx.selfloop_edges(G))
    else:
        raise ValueError("degree_mode must be 'equal', 'unequal', or 'custom'")

    # ---------------------------------------------------------------- add weights
    if weighted:
        for u, v in G.edges():
            G[u][v]["weight"] = weight_sampler()
            

    # guarantee reproducible ordering of nodes
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    return G

def calculate_edge_size(
        n: int,     # node size
        directed: bool   # is directed             
        ):
    edge_low = n + n
    edge_high = n * (n - 2) if directed else (n * (n - 2) // 2)
    return edge_low, edge_high

if __name__ == "__main__":
    base_dir = "../../evaluation_dataset"
    node_size_list = [10, 50, 95, 105, 3500, 7000]
    weighted = [True, False]
    degree_modes = ["equal", "unequal"]
    node_edge_dict = {}
    for node_size in node_size_list:
        node_edge_dict[node_size] = calculate_edge_size(node_size, directed=True)
    for key, values in node_edge_dict.items():
        # if key == 10 or key == 50 or key == 95:
        #     for i in range(100):
        #         for edge_num in values:
        #             for degree in degree_modes:
        #                 for w in weighted:
        #                     store_path = os.path.join(base_dir, f"node_size_{key}",str(f"edge_num_{edge_num}"), degree, "directed")
        #                     os.makedirs(store_path, exist_ok=True)
        #                     G = build_graph(n=key, degree_mode=degree, num_edges=edge_num, weighted=w, directed=True)
        #                     if w:
        #                         file_name = f"{i}_n{key}_e{edge_num}_{degree}_graph_weighted.edgelist"
        #                         nx.write_edgelist(G,
        #                         path=os.path.join(store_path, file_name),
        #                         data=["weight"])   
        #                     else:
        #                         file_name = f"{i}_n{key}_e{edge_num}_{degree}_graph.edgelist"
        #                         nx.write_edgelist(G,
        #                         path=os.path.join(store_path, file_name),
        #                         data=False)
        # elif key == 105:
        #     for i in range(30):
        #         for edge_num in values:
        #             for degree in degree_modes:
        #                 for w in weighted:
        #                     store_path = os.path.join(base_dir, f"node_size_{key}",str(f"edge_num_{edge_num}"), degree, "directed")
        #                     os.makedirs(store_path, exist_ok=True)
        #                     G = build_graph(n=key, degree_mode=degree, num_edges=edge_num, weighted=w, directed=True)
        #                     if w:
        #                         file_name = f"{i}_n{key}_e{edge_num}_{degree}_graph_weighted.edgelist"
        #                         nx.write_edgelist(G,
        #                         path=os.path.join(store_path, file_name),
        #                         data=["weight"])   
        #                     else:
        #                         file_name = f"{i}_n{key}_e{edge_num}_{degree}_graph.edgelist"
        #                         nx.write_edgelist(G,
        #                         path=os.path.join(store_path, file_name),
        #                         data=False)
        # else:
        if key == 3500 or key == 7000:
            for i in range(2):
                for edge_num in values:
                    for degree in degree_modes:
                        for w in weighted:
                            store_path = os.path.join(base_dir, f"node_size_{key}",str(f"edge_num_{edge_num}"), degree, "directed")
                            os.makedirs(store_path, exist_ok=True)
                            G = build_graph(n=key, degree_mode=degree, num_edges=edge_num, weighted=w, directed=True)
                            if w:
                                file_name = f"{i}_n{key}_e{edge_num}_{degree}_graph_weighted.edgelist"
                                nx.write_edgelist(G,
                                path=os.path.join(store_path, file_name),
                                data=["weight"])   
                            else:
                                file_name = f"{i}_n{key}_e{edge_num}_{degree}_graph.edgelist"
                                nx.write_edgelist(G,
                                path=os.path.join(store_path, file_name),
                                data=False)
