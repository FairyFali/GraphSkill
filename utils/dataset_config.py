"""
Dataset configuration helper for unified runner scripts.

Centralizes all dataset-specific differences (data loaders, paths, valid versions)
so that each runner script only needs to call get_dataset_config() with a benchmark name.
"""

from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple


def get_dataset_config(benchmark: str, dataset_version: str) -> Dict[str, Any]:
    """
    Return dataset-specific configuration based on benchmark name.

    Args:
        benchmark: One of 'complexgraph' or 'gtools'
        dataset_version: Dataset version (e.g., 'small', 'large', 'composite')

    Returns:
        Dict with keys:
            - load_data: Callable that loads (questions_data, graphs_data)
            - valid_versions: List of valid dataset version strings
            - output_base: Base output directory name (e.g., 'complexgraph' or 'gtools')
            - test_case_path: Path to test cases directory
            - multi_test_case_path: Path to multiple test cases directory
            - label: Human-readable benchmark label
            - benchmark: The benchmark name string

    Raises:
        ValueError: If benchmark or dataset_version is invalid
    """
    configs = {
        "complexgraph": {
            "valid_versions": ["small", "large", "composite"],
            "output_base": "complexgraph",
            "test_case_path": "prompts/graph_tasks_testing_cases/",
            "multi_test_case_path": "prompts/graph_tasks_testing_cases_multiple/",
            "label": "ComplexGraph",
            "benchmark": "complexgraph",
            "docs_repo": "data/networkx_graph_functions_docs.json"
        },
        "gtools": {
            "valid_versions": ["small", "large"],
            "output_base": "gtools",
            "test_case_path": "prompts/GTools_testing_cases/edge_list/",
            "multi_test_case_path": "prompts/GTools_testing_cases/edge_list/",
            "label": "GTools",
            "benchmark": "gtools",
            "docs_repo": "data/networkx_graph_functions_docs.json"
        },
    }

    if benchmark not in configs:
        raise ValueError(f"Unknown benchmark: {benchmark}. Must be one of {list(configs.keys())}")

    config = configs[benchmark]

    if dataset_version not in config["valid_versions"]:
        raise ValueError(
            f"Invalid dataset version '{dataset_version}' for {benchmark}. "
            f"Valid versions: {config['valid_versions']}"
        )

    # Lazy import to avoid circular imports
    if benchmark == "complexgraph":
        from utils.complexgraph_utils import load_complexgraph_data
        config["load_data"] = load_complexgraph_data
    elif benchmark == "gtools":
        from utils.gtools_utils import load_gtools_data
        config["load_data"] = load_gtools_data

    return config
