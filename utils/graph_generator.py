from __future__ import annotations
from operator import is_
import json, random, pathlib, typing as tp
from dataclasses import dataclass, field
from enum import Enum, auto
import math
import networkx as nx
from slugify import slugify
from tqdm import tqdm
from typing import Iterable, Union, List, Tuple, Dict

# ─────────────────────────────────────────────────────────────
# 2.  GRAPH GENERATION  +  3.  VERIFICATION/LABEL
# ─────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Existing record structure (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GraphRecord:
    problem_slug: str
    graph_id: str
    num_nodes: int
    directed: bool
    connected: bool
    cyclic: bool
    label: str              # {c, dc, cy, acy}
    path: pathlib.Path      # where the pickled graph is stored


# ──────────────────────────────────────────────────────────────────────────────
# New generator
# ──────────────────────────────────────────────────────────────────────────────
class GraphGenerator:
    """
    Generate one of eight common graph families.

    graph_cat in {
        "Sparse_Graph", "Planar_Graph", "Regular_Graph", "Dense_Graph",
        "Complete_Graph", "Small_World_Graph", "Erdos_Renyi_Graph", "Power_Law_Graph"
    }

    Returns either:
      - list[(u, v)]                  if weighted=False
      - list[(u, v, w)]               if weighted=True
      - or the NetworkX graph itself  if as_graph=True
    """

    graph_catS = {
        "Sparse_Graph",
        "Planar_Graph",
        "Regular_Graph",
        "Dense_Graph",
        "Complete_Graph",
        "Small_World_Graph",
        "Erdos_Renyi_Graph",
        "Power_Law_Graph",
    }

    @staticmethod
    def generate_graph(
        rng: random.Random,
        graph_cat: str,
        node_range: tuple[int, int] = (20, 200),
        *,
        weighted: bool = False,
        weight_range: tuple[float, float] = (0.0, 10.0),
        weight_mode: str = "uniform",  # "uniform" | "int" | "normal"
        as_graph: bool = False,
        is_large: bool = False,
        is_directed: bool = False,
    ) -> Union[List[Tuple[int, int]], List[Tuple[int, int, float]], nx.Graph]:
        """
        Create a graph of the requested family. By default returns an edge list.
        If weighted=True, each edge gets a 'weight' attribute and the edge-list
        items are (u, v, w). Use as_graph=True to return the NetworkX graph.
        """
        if graph_cat not in GraphGenerator.graph_catS:
            raise ValueError(f"Unknown graph_cat={graph_cat!r}")

        if not is_large:
            n = rng.randint(*node_range)

        if graph_cat == "Sparse_Graph":
            p = rng.uniform(0.03, 0.05)
            if is_large:
                n = 10000
            g = nx.gnp_random_graph(n, p, seed=rng.randint(0, 2**32 - 1))

        elif graph_cat == "Planar_Graph":
            if is_large:
                n = 10000
            m = max(2, int(math.isqrt(n)))
            grid = nx.grid_2d_graph(m, m)
            g = nx.convert_node_labels_to_integers(grid)

        elif graph_cat == "Regular_Graph":
            if is_large:
                n = 5400
            k = rng.randint(2, max(2, min(6, n - 1)))
            if (k * n) % 2 == 1:
                k += 1
            g = nx.random_regular_graph(k, n, seed=rng.randint(0, 2**32 - 1))

        elif graph_cat == "Dense_Graph":
            if is_large:
                n = 225
            g = nx.gnp_random_graph(n, 0.8, seed=rng.randint(0, 2**32 - 1))

        elif graph_cat == "Complete_Graph":
            if is_large:
                n = 200
            g = nx.complete_graph(n)

        elif graph_cat == "Small_World_Graph":
            if is_large:
                n = 6000
            k_sw = max(2, min(n - 1, 6))
            p_sw = rng.uniform(0.1, 0.3)
            g = nx.watts_strogatz_graph(n, k_sw, p_sw, seed=rng.randint(0, 2**32 - 1))

        elif graph_cat == "Erdos_Renyi_Graph":
            if is_large:
                n = 1800
            p = 0.05
            g = nx.gnp_random_graph(n, p, seed=rng.randint(0, 2**32 - 1))

        elif graph_cat == "Power_Law_Graph":
            if is_large:
                n = 8000
            m_ba = rng.randint(2, max(2, min(10, n - 1)))
            g = nx.barabasi_albert_graph(n, m_ba, seed=rng.randint(0, 2**32 - 1))
        else:
            raise RuntimeError("Unhandled graph_cat")

        # --- Optional: add weights -----------------------------------------
        if weighted:
            a, b = weight_range

            def draw_weight():
                if weight_mode == "uniform":
                    return rng.uniform(a, b)
                elif weight_mode == "int":
                    lo, hi = int(math.floor(a)), int(math.ceil(b))
                    return rng.randint(lo, hi)
                elif weight_mode == "normal":
                    # Centered in the range, clipped back to [a, b]
                    mu = (a + b) / 2.0
                    sigma = max(1e-12, (b - a) / 6.0)  # ~99.7% inside [a,b]
                    w = rng.gauss(mu, sigma)
                    return min(b, max(a, w))
                else:
                    raise ValueError(f"Unknown weight_mode={weight_mode!r}")

            for u, v in g.edges():
                g[u][v]["weight"] = float(draw_weight())

        if as_graph:
            return g

        if weighted:
            return [(u, v, d["weight"]) for u, v, d in g.edges(data=True)]
        else:
            return list(g.edges())

    # def convert_to_acyclic(self, g: nx.Graph) -> nx.Graph:
    #     """
    #     Convert a graph to an acyclic graph.
    #     """
    #     return nx.transitive_reduction(g)

    # ------------------------------------------------------------------
    # Helper: rebuild a NetworkX graph from an edge list
    # ------------------------------------------------------------------
    @staticmethod
    def to_networkx(
        edge_list: Iterable[Tuple],
        *,
        directed: bool = False,
    ) -> nx.Graph:
        """
        Build nx.Graph or nx.DiGraph from an edge list that is either
        (u, v) or (u, v, w). Detects weighted vs unweighted automatically.
        """
        G = nx.DiGraph() if directed else nx.Graph()
        edge_list = list(edge_list)
        if not edge_list:
            return G

        if len(edge_list[0]) == 3:
            G.add_weighted_edges_from(edge_list)  # (u, v, w)
        else:
            G.add_edges_from(edge_list)           # (u, v)
        return G

    @staticmethod
    def to_edge_list(
        G: nx.Graph,
        weighted: bool
    ) -> Iterable[Tuple]:
        """
        Convert nx.Graph or nx.DiGraph to an edge list that is either
        (u, v) or (u, v, w). 
        """
        if weighted:
            return [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
        else:
            return list(G.edges())

    @staticmethod
    def to_adj_list(
        G: nx.Graph,
        weighted: bool
    ) -> Dict[Iterable[Tuple]]:
        """
        Convert nx.Graph or nx.DiGraph to an edge list that is either
        (u, v) or (u, v, w). 
        """
        if weighted:
            return {
                str(node): [(str(nbr), data["weight"]) for nbr, data in neighbors.items()] 
                for node, neighbors in G.adjacency()
                }
        else:
            return nx.to_dict_of_lists(G)

    # ------------------------------------------------------------------
    # Verifier (unchanged)
    # ------------------------------------------------------------------
    @staticmethod
    def verify_and_label(g: nx.Graph) -> tuple[bool, bool, str]:
        """
        Determine connectedness, cyclicity, and assign label:
        {c, dc, cy, acy}.
        """
        undirected = g.to_undirected()
        connected = nx.is_connected(undirected)

        cyclic = (
            not nx.is_forest(undirected)
            if not g.is_directed()
            else not nx.is_directed_acyclic_graph(g)
        )

        if connected and not cyclic:
            label = "c"
        elif not connected and not cyclic:
            label = "dc"
        elif connected and cyclic:
            label = "cy"
        else:
            label = "acy"

        return connected, cyclic, label