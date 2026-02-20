"""
Graph Utilities: Graph type properties and mappings

This module provides utilities for converting graph type keys to their
directedness and weightedness properties.

Functions:
    get_graph_properties(graph_type): Get directed/weighted properties
"""

from typing import Dict, Tuple


def get_graph_properties(graph_type: str) -> Tuple[str, str]:
    """
    Convert graph type key to directedness and weightedness properties.

    Args:
        graph_type: Graph type key (e.g., "weighted_undirected_tasks")

    Returns:
        Tuple of (directedness, weightedness) as strings
        - directedness: "directed" or "undirected"
        - weightedness: "weighted" or "unweighted"

    Raises:
        ValueError: If graph_type is not recognized

    Example:
        is_directed, is_weighted = get_graph_properties("weighted_undirected_tasks")
        # Returns: ("undirected", "weighted")
    """
    graph_type_mapping = {
        "weighted_undirected_tasks": ("undirected", "weighted"),
        "unweighted_directed_tasks": ("directed", "unweighted"),
        "weighted_directed_tasks": ("directed", "weighted"),
        "unweighted_undirected_tasks": ("undirected", "unweighted"),
    }

    if graph_type not in graph_type_mapping:
        raise ValueError(f"Invalid graph type: {graph_type}")

    return graph_type_mapping[graph_type]


def get_graph_properties_dict(graph_type: str) -> Dict[str, bool]:
    """
    Convert graph type key to dictionary with boolean properties.

    Args:
        graph_type: Graph type key (e.g., "weighted_undirected_tasks")

    Returns:
        Dictionary with 'directed' and 'weighted' boolean keys

    Example:
        props = get_graph_properties_dict("weighted_undirected_tasks")
        # Returns: {"directed": False, "weighted": True}
    """
    is_directed, is_weighted = get_graph_properties(graph_type)
    return {
        "directed": is_directed == "directed",
        "weighted": is_weighted == "weighted"
    }
