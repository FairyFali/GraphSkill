"""
TF-IDF Documentation Retrieval Module

This module provides TF-IDF (Term Frequency-Inverse Document Frequency) based
document retrieval functionality for finding relevant documentation snippets
from a structured knowledge repository.

Key Components:
- Repository loading and flattening
- TF-IDF vectorization with cosine similarity
- Top-k document retrieval
- Function name extraction

Example:
    from utils.tfidf_retrieval import find_best_docstring

    docs = find_best_docstring(
        repo_json_path="path/to/docs.json",
        query_text="How to find shortest path in a graph?",
        top_k=5
    )
"""

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


def find_best_docstring(
    repo_json_path: str,
    query_text: str,
    top_k: int = 1
) -> List[str]:
    """
    Retrieve the top-k most relevant documentation snippets using TF-IDF.

    This function performs the following steps:
    1. Load and flatten the documentation repository
    2. Build TF-IDF vectors for all unique docstrings
    3. Build TF-IDF vector for the query text
    4. Compute cosine similarity between query and all docstrings
    5. Return the top-k most similar docstrings

    Args:
        repo_json_path: Path to the JSON documentation repository
        query_text: Query string describing the information need
        top_k: Number of top results to return (default: 1)

    Returns:
        List of the top-k most relevant docstrings, ordered by relevance
        (highest relevance first)

    Raises:
        FileNotFoundError: If the documentation repository doesn't exist
        ValueError: If top_k is less than 1 or greater than number of docs

    Example:
        >>> docs = find_best_docstring(
        ...     repo_json_path="docs/networkx_docs.json",
        ...     query_text="How to compute shortest path between two nodes?",
        ...     top_k=3
        ... )
        >>> print(f"Found {len(docs)} relevant documents")
        Found 3 relevant documents

    Technical Details:
        - Uses TfidfVectorizer with L2 normalization
        - Filters English stop words
        - Cosine similarity metric for ranking
        - Deduplicates docstrings before vectorization
    """
    # Step 1: Load and flatten the repository
    repo = load_repository(repo_json_path)
    docs = flatten_repo(repo)

    # Step 2: Extract unique docstrings (remove duplicates)
    # Using set() ensures we don't process the same docstring multiple times
    docstrings = list(set(docs.values()))

    if not docstrings:
        raise ValueError(f"No docstrings found in repository: {repo_json_path}")

    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    if top_k > len(docstrings):
        print(f"Warning: top_k ({top_k}) is greater than number of documents ({len(docstrings)}). "
              f"Returning all {len(docstrings)} documents.")
        top_k = len(docstrings)

    # Step 3: Build TF-IDF matrix
    # - norm='l2': Normalize feature vectors to unit length (L2 norm)
    # - stop_words='english': Filter common English words (the, is, at, etc.)
    vec = TfidfVectorizer(norm='l2', stop_words='english')

    # Fit vectorizer on all docstrings + query (query is last row)
    # This ensures the query uses the same vocabulary as the documents
    tfidf_matrix = vec.fit_transform(docstrings + [query_text])

    # Step 4: Compute cosine similarity
    # Compare the query vector (last row) with all document vectors
    # cosine_similarity returns a 2D array, flatten to 1D
    cos_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Step 5: Get top-k matches
    # argsort() returns indices that would sort the array
    # [::-1] reverses to get descending order (highest similarity first)
    # [:top_k] takes only the top k results
    top_k_indices = cos_similarities.argsort()[::-1][:top_k]

    # Return the docstrings corresponding to top-k indices
    return [docstrings[i] for i in top_k_indices]


def find_best_function_page(
    repo_json_path: str,
    query_text: str,
    top_k: int = 50
) -> List[str]:
    """
    Retrieve the top-k most relevant function names using TF-IDF.

    Similar to find_best_docstring(), but returns function names instead of
    full docstrings. This is useful for identifying which functions are most
    relevant to a query without retrieving their full documentation.

    Args:
        repo_json_path: Path to the JSON documentation repository
        query_text: Query string describing the information need
        top_k: Number of top function names to return (default: 50)

    Returns:
        List of function names for the top-k most relevant documents,
        ordered by relevance (highest relevance first)

    Raises:
        FileNotFoundError: If the documentation repository doesn't exist
        ValueError: If top_k is less than 1

    Example:
        >>> functions = find_best_function_page(
        ...     repo_json_path="docs/networkx_docs.json",
        ...     query_text="shortest path algorithms",
        ...     top_k=10
        ... )
        >>> print(functions[:3])
        ['shortest_path', 'dijkstra_path', 'bellman_ford_path']

    Note:
        - If a docstring appears multiple times with different function names,
          only one function name will be returned per docstring
        - The mapping from docstring to function name uses the first match found
    """
    # Step 1: Load and flatten the repository
    repo = load_repository(repo_json_path)
    docs = flatten_repo(repo)

    # Step 2: Extract unique docstrings
    docstrings = list(set(docs.values()))

    if not docstrings:
        raise ValueError(f"No docstrings found in repository: {repo_json_path}")

    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    # Allow top_k to exceed document count (will just return all documents)
    actual_k = min(top_k, len(docstrings))

    # Step 3: Build TF-IDF matrix (same as find_best_docstring)
    vec = TfidfVectorizer(norm='l2', stop_words='english')
    tfidf_matrix = vec.fit_transform(docstrings + [query_text])

    # Step 4: Compute cosine similarity
    cos_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Step 5: Get top-k docstrings
    top_k_indices = cos_similarities.argsort()[::-1][:actual_k]
    top_docstrings = [docstrings[i] for i in top_k_indices]

    # Step 6: Map docstrings back to function names
    # For each top-ranked docstring, find the corresponding function name
    function_names = []
    for docstring in top_docstrings:
        # Find the first function name that has this docstring
        for func_name, func_doc in docs.items():
            if func_doc == docstring:
                function_names.append(func_name)
                break  # Only take first match per docstring

    return function_names


# ============================================================================
# Example Usage (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the TF-IDF retrieval functionality.

    To run this example:
        python -m utils.tfidf_retrieval
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.tfidf_retrieval <path_to_docs.json>")
        sys.exit(1)

    repo_path = sys.argv[1]

    # Example query
    query = "How to find the shortest path between two nodes in a graph?"

    print(f"Query: {query}\n")

    # Retrieve top 3 docstrings
    print("Top 3 most relevant docstrings:")
    try:
        docs = find_best_docstring(repo_path, query, top_k=3)
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. {doc[:200]}...")  # Print first 200 chars
    except Exception as e:
        print(f"Error: {e}")

    # Retrieve top 5 function names
    print("\n\nTop 5 most relevant function names:")
    try:
        functions = find_best_function_page(repo_path, query, top_k=5)
        for i, func in enumerate(functions, 1):
            print(f"{i}. {func}")
    except Exception as e:
        print(f"Error: {e}")
