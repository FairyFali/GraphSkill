

TASK_NX_FUNC = {'clustering_and_shortest_path': ['clustering', 'shortest_path_length'], 'highest_clustered_node_in_shortest_path': ['clustering', 'all_shortest_paths'], 'largest_component_and_diameter': ['is_connected', 'diameter', 'connected_components'], 'shortest_path_and_eular_tour': ['has_eulerian_path', 'diameter', 'clustering'], 'pair_tightness_score': ['shortest_path_length', 'common_neighbors', 'clustering'], 'flow_aware_local_clustering_among_cc': ['connected_components', 'maximum_flow', 'clustering'], 'endpoint_aware_flow_score': ['maximum_flow', 'clustering'], 'scc_and_diameter_nested': ['strongly_connected_components', 'diameter'], 'scc_and_eulerian_feasibility': ['has_eulerian_path', 'strongly_connected_components'], 'strongly_connected_components': ['strongly_connected_components'], 'maximum_flow': ['maximum_flow'], 'is_bipartite': ['is_bipartite'], 'find_cycle': ['find_cycle'], 'shortest_path_length': ['shortest_path_length'], 'is_regular': ['is_regular'], 'enumerate_all_cliques': ['enumerate_all_cliques'], 'has_node': ['has_node'], 'is_distance_regular': ['is_distance_regular'], 'has_eulerian_path': ['has_eulerian_path'], 'degree': ['degree'], 'number_of_edges': ['number_of_edges'], 'has_edge': ['has_edge'], 'number_of_nodes': ['number_of_nodes'], 'clustering': ['clustering'], 'is_connected': ['is_connected'], 'connected_components': ['connected_components'], 'diameter': ['diameter'], 'maximum_independent_set': ['maximum_independent_set'], 'topological_sort': ['topological_sort'], 'has_path': ['has_path'], 'max_clique': ['max_clique'], 'common_neighbors': ['common_neighbors']}


import json
from typing import Dict, List, Any, Tuple
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_repository(json_path: str) -> Dict[str, Any]:
    """
    Load a documentation repository from a JSON file.

    Args:
        json_path: Path to the JSON file containing the documentation repository

    Returns:
        Dictionary containing the nested documentation structure

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON

    Example:
        >>> repo = load_repository("docs/networkx_docs.json")
        >>> print(type(repo))
        <class 'dict'>
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Documentation repository not found: {json_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in documentation repository: {json_path}",
            e.doc,
            e.pos
        )


def flatten_repo(repo: Dict[str, Any]) -> Dict[str, str]:
    """
    Walk a repository with nested dicts/lists and extract all docstrings.

    This function handles two common patterns in documentation repositories:

    Pattern 1 - Category or subcategory:
        ["summary", {...nested...}, {...nested...}]

    Pattern 2 - Function definition:
        ["summary", "full docstring"]

    Args:
        repo: Nested dictionary/list structure representing the documentation

    Returns:
        Dictionary mapping function names to their full docstrings

    Example:
        >>> repo = {"algorithms": ["Algorithms", {"shortest_path": ["Summary", "Full docs"]}]}
        >>> docs = flatten_repo(repo)
        >>> print(docs.keys())
        dict_keys(['shortest_path'])

    Note:
        - Paths are constructed as "Top-level > Subcategory > function"
        - Only string docstrings are extracted (non-string values ignored)
        - Duplicate function names will be overwritten by last occurrence
    """
    docs = {}

    def dfs(node: Any, cur_path: List[str]) -> None:
        """
        Depth-first search through nested structure to extract docstrings.

        Args:
            node: Current node in the tree (can be dict, list, or primitive)
            cur_path: Current path from root to this node (list of keys)
        """
        # Case 1: Dictionary node - descend into all key-value pairs
        if isinstance(node, dict):
            for key, val in node.items():
                dfs(val, cur_path + [key])

        # Case 2: List node - handle two patterns
        elif isinstance(node, list):
            # Pattern 1: Category or subcategory with nested children
            # Format: ["summary", {...nested...}, {...nested...}]
            if len(node) >= 2 and isinstance(node[0], str):
                # Descend into anything nested after the summary
                for child in node[1:]:
                    dfs(child, cur_path)

            # Pattern 2: Function definition with docstring
            # Format: ["summary", "full docstring"]
            if len(node) == 2 and isinstance(node[1], str):
                # Last element in current path is the function name
                func_name = cur_path[-1] if cur_path else "unknown"
                docs[func_name] = node[1]

        # Case 3: Primitive types (str, int, etc.) - ignore
        # These are typically summaries or metadata, not docstrings

    # Start DFS from the root with empty path
    dfs(repo, [])
    return docs

repo_json_path = "data/networkx_graph_functions_docs.json"
repo = load_repository(repo_json_path)
flattened_docs = flatten_repo(repo)

ret_gt = {}
for taskname in TASK_NX_FUNC:
    func_list = TASK_NX_FUNC[taskname]
    ret_gt[taskname] = []
    for func in func_list:
        if func in flattened_docs:
            doc = flattened_docs[func]
            ret_gt[taskname].append(doc)
        else: 
            for key in flattened_docs:
                if func.lower() in key.lower():
                    doc = flattened_docs[key]
                    ret_gt[taskname].append(doc)
with open("data/retrieval_groundtruth.json", "w", encoding="utf-8") as f:
    json.dump(ret_gt, f, indent=2)

print("Retrieval Groundtruth generation completed. Saved to data/retrieval_groundtruth.json")