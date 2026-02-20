"""
graph_eval_pipeline.py
Replicates the Figure‑2 pipeline used to build the GraphEva136K dataset.
─────────────────────────────────────────────────────────────────────────────
Dependencies
------------
pip install networkx==3.* tqdm requests beautifulsoup4 python-slugify
─────────────────────────────────────────────────────────────────────────────
High‑level Flow
---------------
1. DATA COLLECTION
   • Fetch metadata for LeetCode contest problems
   • Bucket by difficulty and random‑sample a user‑specified number/ratio
2. GRAPH GENERATION
   • For every sampled problem, call its reference solution (or placeholder)
     to decide the node‑count range, then create a NetworkX graph
   • Verify structural properties (connectivity & cyclicity)
3. LABEL / CLUSTER
   • Assign one of four labels:  c, dc, cy, acy
   • Save graphs and an index in JSONL / pickled NetworkX
"""

import json
import random
import pathlib
from collections import defaultdict
from typing import Dict, Hashable, Optional
import networkx as nx
from utils.graph_generator import GraphGenerator
import argparse

def get_graph_output_base_dir(node_size: str):
    if node_size == "small":
        output_dir = pathlib.Path("/data/chenglin/small_graphtutor_graph_dataset")
    elif node_size == "medium":
        output_dir = pathlib.Path("/data/chenglin/graphtutor_graph_dataset")
    elif node_size == "large":
        output_dir = pathlib.Path("/data/chenglin/large_graphtutor_graph_dataset")
    else:
        raise ValueError(f"unsupported node size {node_size}")
    return output_dir


def save_dict_to_json(data: dict, file_path: str | pathlib.Path, *, indent: int = 4) -> None:
    """
    Serialize a Python dictionary to a JSON file, creating any
    intermediate directories that don’t yet exist.

    Parameters
    ----------
    data : dict
        The dictionary you want to save.
    file_path : str | Path
        Destination path (e.g. "output/settings/config.json").
    indent : int, optional
        Pretty-print spacing in the resulting file (default = 4).
    """
    file_path = pathlib.Path(file_path)                 # allow strings or Path objects

    # Ensure parent directories exist (no error if they’re already there)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the JSON file with UTF-8 encoding
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def generate_graphs_for_category(graph_gen, rng, graph_category, graph_type, node_size, target_per_label=10):
    """
    Generate graphs for a specific category and type until all labels reach target.
    
    Args:
        graph_gen: GraphGenerator instance
        rng: Random number generator
        graph_category: Type of graph (e.g., "Sparse_Graph", "Planar_Graph")
        graph_type: Task type (e.g., "weighted_undirected_tasks")
        target_per_label: Number of graphs to generate per label
        
    Returns:
        dict: Dictionary mapping labels to lists of edge lists
    """
    graphs_by_label = {'c': [], 'dc': [], 'cy': [], 'acy': []}
    
    print(f"  Generating {graph_category} for {graph_type}...")
    
    # Determine if graphs should be directed and weighted based on task type
    is_directed = not "undirected" in graph_type
    is_weighted = not "unweighted" in graph_type
    
    attempts = 0
    max_attempts = target_per_label * 100  # Prevent infinite loops
    # import pdb; pdb.set_trace()
    if node_size == "small":
        node_range = (3, 4)
        is_large = False
    elif node_size == "medium":
        node_range = (5, 200)
        is_large = False
    elif node_size == "large":
        node_range = (5000, 10000)
        is_large=True
    else:
        raise ValueError(f"unsupported node size {node_size}")
    
    while (any(len(v) < target_per_label for v in graphs_by_label.values())) and attempts < max_attempts:
        attempts += 1
        # import pdb; pdb.set_trace()
        # Generate graph with appropriate parameters
        edge_list = graph_gen.generate_graph(
            rng=rng, 
            node_range=node_range,
            graph_cat=graph_category,
            weighted=is_weighted,
            weight_range=(0.0, 10.0),
            weight_mode="uniform",
            is_large=is_large,
            is_directed=is_directed
        )
        
        if len(edge_list) == 0:
            continue
        
        # Convert to NetworkX graph for verification
        G = graph_gen.to_networkx(edge_list, directed=is_directed)
        
        
        # Verify and label the graph
        connected, cyclic, label = graph_gen.verify_and_label(G)
        
        # Store only if this label still needs more samples
        if len(graphs_by_label[label]) < target_per_label:
            graphs_by_label[label].append(edge_list)
        
        
        added_label = [label for label in list(graphs_by_label.keys()) if len(graphs_by_label[label])>= target_per_label]
        
        if attempts % 1000 == 0:
            print(f"    Attempts: {attempts}, Added Labels: {added_label}")
    
    print(f"    Final results for {graph_category}: {added_label}")
    return graphs_by_label

def main(node_size):
    """Main function to generate graphs for all categories and types."""
    output_dir = get_graph_output_base_dir(node_size)
    # Configuration
    TARGET_PER_LABEL = 100  # Reduced for testing, increase for production
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    # Graph categories available
    all_graph_categories = [
        "Sparse_Graph", "Planar_Graph", "Regular_Graph", "Dense_Graph", 
        "Complete_Graph", "Small_World_Graph", "Erdos_Renyi_Graph", "Power_Law_Graph"
    ]
    direct_graph_categories = [
        "Sparse_Graph", "Planar_Graph", "Regular_Graph", "Dense_Graph", "Small_World_Graph", "Erdos_Renyi_Graph", "Power_Law_Graph"
    ]
    
    # Task types and their properties
    all_tasks = {
        "weighted_undirected_tasks": {
            "directed": False,
            "weighted": True
        },
        "unweighted_directed_tasks": {
            "directed": True,
            "weighted": False
        },
        "weighted_directed_tasks": {
            "directed": True,
            "weighted": True
        },
        "unweighted_undirected_tasks": {
            "directed": False,
            "weighted": False
        }
    }
    
    # Initialize graph generator
    graph_gen = GraphGenerator()
    
    # Output directory
    
    
    print("Starting graph generation...")
    print(f"Target per label: {TARGET_PER_LABEL}")
    print(f"Graph categories: {len(all_graph_categories)}")
    print(f"Task types: {len(all_tasks)}")
    
    # Generate graphs for each task type
    for graph_type, config in all_tasks.items():
        print(f"\n{'='*10}")
        print(f"Processing: {graph_type}")
        print(f"Properties: directed={config['directed']}, weighted={config['weighted']}")
        print('='*10)
        if graph_type == "weighted_undirected_tasks":
            is_directed = "undirected"
            is_weighted = "weighted"
        elif graph_type == "unweighted_directed_tasks":
            is_directed = "directed"
            is_weighted = "unweighted"
        elif graph_type == "weighted_directed_tasks":
            is_directed = "directed"
            is_weighted = "weighted"
        elif graph_type == "unweighted_undirected_tasks":
            is_directed = "undirected"
            is_weighted = "unweighted"
        else:
            raise ValueError(f"Invalid graph type: {graph_type}")
        
        if config['directed']:
            graph_categories = direct_graph_categories
        else:
            graph_categories = all_graph_categories
        
        for graph_category in graph_categories:
            print(f"\n  Category: {graph_category}")
            
            # Generate graphs for this category
            graphs_by_label = generate_graphs_for_category(
                graph_gen, rng, graph_category, graph_type, node_size, TARGET_PER_LABEL
            )
            # Save results
            for key in [k for k, v in graphs_by_label.items() if not v]:
                del graphs_by_label[key]
                
            output_path = output_dir / is_directed/ is_weighted / f"{graph_category}.json"
            
            save_dict_to_json(graphs_by_label, output_path)
            print(f"    Saved to: {output_path}")

    print(f"\n{'='*10}")
    print("Graph generation completed!")
    print(f"Output directory: {output_dir}")
    print('='*10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that generates graphs by different .")
    parser.add_argument("--node_size", type=str, help="size of node, enter in string small, medium, large")
    args = parser.parse_args()
    main(args.node_size)

    # main()                

                