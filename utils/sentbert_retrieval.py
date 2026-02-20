"""
Sentence-BERT Documentation Retrieval Module

This module provides semantic similarity-based document retrieval using
Sentence-BERT (SBERT) embeddings. Unlike TF-IDF which uses word frequency,
SBERT captures semantic meaning and context, enabling better retrieval of
conceptually related documentation.

Key Features:
- Semantic similarity using pre-trained SBERT model
- L2-normalized embeddings for efficient cosine similarity
- GPU acceleration support (if available)
- Top-k document retrieval with similarity scores

Example:
    from utils.sentbert_retrieval import find_best_docstring

    docs = find_best_docstring(
        repo_json_path="path/to/docs.json",
        query_text="How to find shortest path in a graph?",
        top_k=5
    )

Technical Details:
    - Model: sentence-transformers/all-MiniLM-L6-v2
    - Embedding dimension: 384
    - Similarity metric: Cosine similarity (dot product of normalized vectors)
    - The model is loaded once at module import to avoid overhead
"""

import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
import torch


# ============================================================================
# Global Model Instance
# ============================================================================

# Load the Sentence-BERT model once at module import
# This avoids re-loading the model on every function call (significant overhead)
# Model: all-MiniLM-L6-v2 - 384-dimensional embeddings, ~80MB
# Alternative models:
#   - all-mpnet-base-v2: Higher quality, 768-dim, ~420MB
#   - paraphrase-MiniLM-L3-v2: Faster, 384-dim, ~60MB
_MODEL: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """
    Lazy-load the Sentence-BERT model on first use.

    Returns:
        Pre-trained SentenceTransformer model

    Note:
        The model is cached in the global _MODEL variable to avoid
        reloading on subsequent calls.
    """
    global _MODEL
    if _MODEL is None:
        print("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✓ Model loaded successfully")
    return _MODEL


# ============================================================================
# Repository Loading and Flattening
# ============================================================================

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


def flatten_repo(repo: Dict[str, Any]) -> List[Tuple[str, str]]:
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
        List of (path, docstring) tuples where:
        - path: Hierarchical location like "Top-level > Subcategory > function"
        - docstring: Full documentation string for the function

    Example:
        >>> repo = {"algorithms": ["Algorithms", {"shortest_path": ["Summary", "Full docs"]}]}
        >>> docs = flatten_repo(repo)
        >>> print(docs[0])
        ('algorithms > shortest_path', 'Full docs')

    Note:
        - Returns list of tuples (unlike TF-IDF which returns dict)
        - Preserves path information for better context
        - Handles arbitrarily nested structures
    """
    docs = []

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
                # Create hierarchical path
                path_str = " > ".join(cur_path) if cur_path else "unknown"
                docs.append((path_str, node[1]))

        # Case 3: Primitive types (str, int, etc.) - ignore
        # These are typically summaries or metadata, not docstrings

    # Start DFS from the root with empty path
    dfs(repo, [])
    return docs


# ============================================================================
# Sentence-BERT Retrieval
# ============================================================================

def find_best_docstring(
    repo_json_path: str,
    query_text: str,
    top_k: int = 1,
    return_scores: bool = False
) -> List[str]:
    """
    Retrieve the top-k most semantically similar docstrings using Sentence-BERT.

    This function performs semantic similarity search using pre-trained Sentence-BERT
    embeddings. Unlike TF-IDF (which matches keywords), SBERT captures semantic
    meaning, enabling better retrieval of conceptually related documentation.

    Process:
    1. Load and flatten the documentation repository
    2. Encode all docstrings into 384-dimensional embeddings (cached)
    3. Encode the query text into an embedding
    4. Compute cosine similarity (dot product of normalized vectors)
    5. Return top-k most similar docstrings

    Args:
        repo_json_path: Path to the JSON documentation repository
        query_text: Query string describing the information need
        top_k: Number of top results to return (default: 1)
        return_scores: If True, return (docstring, score) tuples instead of just docstrings

    Returns:
        If return_scores=False: List of docstrings ordered by relevance
        If return_scores=True: List of (docstring, score) tuples

    Raises:
        FileNotFoundError: If the documentation repository doesn't exist
        ValueError: If top_k is less than 1 or repository is empty

    Example:
        >>> docs = find_best_docstring(
        ...     repo_json_path="docs/networkx_docs.json",
        ...     query_text="How to compute shortest path between two nodes?",
        ...     top_k=3
        ... )
        >>> print(f"Found {len(docs)} relevant documents")
        Found 3 relevant documents

        >>> # Get results with similarity scores
        >>> results = find_best_docstring(
        ...     repo_json_path="docs/networkx_docs.json",
        ...     query_text="graph clustering algorithm",
        ...     top_k=5,
        ...     return_scores=True
        ... )
        >>> for doc, score in results:
        ...     print(f"Score: {score:.3f} - {doc[:100]}...")

    Technical Details:
        - Model: all-MiniLM-L6-v2 (384-dimensional embeddings)
        - Embeddings are L2-normalized for efficient cosine similarity
        - Cosine similarity = dot product of normalized vectors
        - Similarity scores range from -1 (opposite) to 1 (identical)
        - GPU acceleration used automatically if available
    """
    # Step 1: Load and flatten the repository
    repo = load_repository(repo_json_path)
    docs = flatten_repo(repo)

    if not docs:
        raise ValueError(f"No docstrings found in repository: {repo_json_path}")

    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    # Step 2: Extract unique docstrings
    # Note: We keep the paths for reference but deduplicate docstrings
    paths, docstrings = zip(*docs)
    unique_docstrings = list(set(docstrings))

    if top_k > len(unique_docstrings):
        print(f"Warning: top_k ({top_k}) is greater than number of unique documents "
              f"({len(unique_docstrings)}). Returning all documents.")
        top_k = len(unique_docstrings)

    # Step 3: Get the Sentence-BERT model
    model = _get_model()

    # Step 4: Encode corpus (all docstrings) into embeddings
    # - convert_to_tensor=True: Return PyTorch tensors (faster operations)
    # - normalize_embeddings=True: L2-normalize vectors (cosine similarity = dot product)
    # This creates a tensor of shape [num_docs, 384]
    corpus_embeddings = model.encode(
        unique_docstrings,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Step 5: Encode query into embedding
    # This creates a tensor of shape [384]
    query_embedding = model.encode(
        query_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Step 6: Compute cosine similarity
    # For normalized vectors: cosine_similarity(a, b) = dot_product(a, b)
    # util.dot_score returns shape [1, num_docs], squeeze to [num_docs]
    similarities = util.dot_score(query_embedding, corpus_embeddings).squeeze(0)

    # Step 7: Get top-k indices
    # torch.topk returns (values, indices) sorted in descending order
    top_k_actual = min(top_k, len(similarities))
    top_results = torch.topk(similarities, k=top_k_actual)
    top_indices = top_results.indices.tolist()
    top_scores = top_results.values.tolist()

    # Step 8: Return results
    if return_scores:
        return [
            (unique_docstrings[idx], score)
            for idx, score in zip(top_indices, top_scores)
        ]
    else:
        return [unique_docstrings[idx] for idx in top_indices]


def find_best_function_page(
    repo_json_path: str,
    query_text: str,
    top_k: int = 50
) -> List[str]:
    """
    Retrieve the top-k most relevant function names using Sentence-BERT.

    Similar to find_best_docstring(), but returns function names (paths) instead
    of full docstrings. This is useful for identifying which functions are most
    relevant without retrieving their full documentation.

    Args:
        repo_json_path: Path to the JSON documentation repository
        query_text: Query string describing the information need
        top_k: Number of top function names to return (default: 50)

    Returns:
        List of hierarchical function paths (e.g., "algorithms > shortest_path")
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
        ['algorithms > shortest_path > dijkstra_path',
         'algorithms > shortest_path > bellman_ford_path',
         'algorithms > shortest_path > astar_path']

    Note:
        - Returns paths (not just function names) for better context
        - If a docstring appears multiple times, only first path is returned
        - Paths use " > " separator for hierarchical structure
    """
    # Step 1: Load and flatten the repository
    repo = load_repository(repo_json_path)
    docs = flatten_repo(repo)

    if not docs:
        raise ValueError(f"No docstrings found in repository: {repo_json_path}")

    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    # Step 2: Extract paths and docstrings
    paths_list, docstrings_list = zip(*docs)

    # Step 3: Deduplicate docstrings while preserving path mapping
    unique_docstrings = list(set(docstrings_list))
    actual_k = min(top_k, len(unique_docstrings))

    # Step 4: Get the Sentence-BERT model
    model = _get_model()

    # Step 5: Encode corpus and query
    corpus_embeddings = model.encode(
        unique_docstrings,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = model.encode(
        query_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Step 6: Compute cosine similarity
    similarities = util.dot_score(query_embedding, corpus_embeddings).squeeze(0)

    # Step 7: Get top-k indices
    top_results = torch.topk(similarities, k=actual_k)
    top_indices = top_results.indices.tolist()

    # Step 8: Map docstrings back to function paths
    # For each top-ranked docstring, find the first matching path
    top_docstrings = [unique_docstrings[idx] for idx in top_indices]

    function_paths = []
    for docstring in top_docstrings:
        # Find the first path that has this docstring
        for path, doc in docs:
            if doc == docstring:
                function_paths.append(path)
                break  # Only take first match per docstring

    return function_paths


# ============================================================================
# Example Usage (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the Sentence-BERT retrieval functionality.

    To run this example:
        python -m utils.sentbert_retrieval <path_to_docs.json>
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.sentbert_retrieval <path_to_docs.json>")
        sys.exit(1)

    repo_path = sys.argv[1]

    # Example query
    query = "How to find the shortest path between two nodes in a graph?"

    print(f"Query: {query}\n")

    # Retrieve top 3 docstrings with scores
    print("Top 3 most relevant docstrings (with similarity scores):")
    try:
        results = find_best_docstring(repo_path, query, top_k=3, return_scores=True)
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. Similarity: {score:.4f}")
            print(f"   {doc[:200]}...")  # Print first 200 chars
    except Exception as e:
        print(f"Error: {e}")

    # Retrieve top 5 function paths
    print("\n\nTop 5 most relevant function paths:")
    try:
        functions = find_best_function_page(repo_path, query, top_k=5)
        for i, func in enumerate(functions, 1):
            print(f"{i}. {func}")
    except Exception as e:
        print(f"Error: {e}")
