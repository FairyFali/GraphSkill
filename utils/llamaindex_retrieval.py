"""
LlamaIndex-based Documentation Retrieval Utilities

This module provides vector-based document retrieval using LlamaIndex,
designed for the GraphTeam method and other RAG approaches. It supports
both on-the-fly retrieval with embeddings and loading pre-retrieved
(cached) documentation.

Key Features:
- Vector similarity retrieval with LlamaIndex
- Repository parsing (nested dict/list structures)
- Similarity threshold filtering
- Pre-retrieved documentation caching
- Fallback handling when LlamaIndex unavailable

Example Usage:
    from utils.llamaindex_retrieval import retrieve_docs_with_vector_store

    # On-the-fly retrieval
    docs = retrieve_docs_with_vector_store(
        docs_repo_path=Path("networkx_docs.json"),
        task_description="Find shortest path in a graph",
        top_k=5,
        similarity_threshold=0.70
    )

    # Load cached docs
    from utils.llamaindex_retrieval import load_pre_retrieved_docs
    cached_docs = load_pre_retrieved_docs(
        docs_file_path=Path("cached_docs.json"),
        task_name="clustering"
    )

Cost Analysis (OpenAI Embedding API):
    By default, LlamaIndex uses OpenAI's text-embedding-ada-002 (cloud API) for embeddings.
    No local embedding model is configured in this module.

    - retrieve_docs_with_vector_store(): Re-embeds ALL documents on every call.
        * Each call embeds ~N documents + 1 query via OpenAI API
        * For our NetworkX docs repo (~300 functions), this is ~300 embedding API calls per task
        * With 22 tasks: ~6,600 embedding API calls total
        * Cost: ~$0.01-0.02 per run (ada-002 is $0.0001/1K tokens)

    - build_retriever() + retrieve_with_retriever(): More efficient pattern.
        * build_retriever() embeds all documents ONCE (~300 API calls)
        * retrieve_with_retriever() embeds only the query (1 API call per task)
        * With 22 tasks: ~322 embedding API calls total (~20x cheaper)

    To run fully locally (no API cost), configure a local embedding model:
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Dependencies:
    - llama-index: pip install llama-index (for vector retrieval)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Try to import LlamaIndex (optional dependency)
try:
    from llama_index.core import VectorStoreIndex, Document
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

# OpenAI embedding API pricing (per 1K tokens)
# text-embedding-ada-002: $0.0001 / 1K tokens (LlamaIndex default)
# text-embedding-3-small: $0.00002 / 1K tokens
# text-embedding-3-large: $0.00013 / 1K tokens
EMBEDDING_COST_PER_1K_TOKENS = 0.0001  # ada-002 default


def _estimate_embedding_cost(texts: list, label: str = "", verbose: bool = True) -> float:
    """Estimate and print the OpenAI embedding API cost for a list of texts."""
    total_chars = sum(len(t) for t in texts)
    total_tokens = total_chars // 4  # ~4 chars per token
    cost = (total_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
    if verbose:
        print(f"  [API Cost] {label}")
        print(f"             Texts: {len(texts)}, Est. tokens: {total_tokens:,}, Est. cost: ${cost:.6f}")
        print(f"             (Using OpenAI text-embedding-ada-002 @ ${EMBEDDING_COST_PER_1K_TOKENS}/1K tokens)")
    return cost


# ============================================================================
# Repository Loading and Parsing
# ============================================================================

def load_repository(json_path: str) -> Dict[str, Any]:
    """
    Load NetworkX documentation repository from JSON file.

    This loads the hierarchically structured NetworkX documentation that
    contains function names, descriptions, and full docstrings.

    Args:
        json_path: Path to the NetworkX documentation JSON file
            Expected format: Nested dict/list structure with function docs

    Returns:
        Dictionary containing the nested documentation structure

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON

    Example:
        >>> repo = load_repository("crawl_documentation/networkx_graph_functions_docs.json")
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
    Walk a nested repository structure and extract all docstrings.

    The NetworkX documentation is organized in a nested dict/list structure:
    - Categories contain subcategories
    - Subcategories contain functions
    - Functions have ["summary", "full_docstring"] format

    This function walks the structure recursively and extracts
    (function_name, docstring) pairs for all documented functions.

    Args:
        repo: Nested dictionary/list structure from NetworkX docs JSON

    Returns:
        List of (function_name, docstring) tuples

    Example:
        >>> repo = {"algorithms": ["Algorithms", {"shortest_path": ["Summary", "Full docs"]}]}
        >>> docs = flatten_repo(repo)
        >>> print(docs[0])
        ('shortest_path', 'Full docs')

    Note:
        Handles two patterns in the documentation structure:
        - Pattern 1: ["summary", {...nested...}] - Category/subcategory
        - Pattern 2: ["summary", "full docstring"] - Function definition
    """
    docs = []

    def dfs(node: Any, cur_path: List[str]) -> None:
        """
        Depth-first search to extract docstrings from nested structure.

        Args:
            node: Current node (can be dict, list, or primitive)
            cur_path: Path from root to current node (list of keys)
        """
        if isinstance(node, dict):
            # Dictionary node: descend into all key-value pairs
            for key, val in node.items():
                dfs(val, cur_path + [key])

        elif isinstance(node, list):
            # Pattern 1: Category or subcategory with nested children
            # Format: ["summary", {...nested...}, {...nested...}]
            if len(node) >= 2 and isinstance(node[0], str):
                # Descend into all children after the summary
                for child in node[1:]:
                    dfs(child, cur_path)

            # Pattern 2: Function definition with docstring
            # Format: ["summary", "full docstring"]
            if len(node) == 2 and isinstance(node[1], str):
                # Last element in current path is the function name
                func_name = cur_path[-1] if cur_path else "unknown"
                docstring = node[1]  # Full docstring
                docs.append((func_name, docstring))

        # Primitive types (str, int, etc.) are ignored
        # These are typically summaries or metadata

    # Start DFS traversal from the root
    dfs(repo, [])
    return docs


# ============================================================================
# Vector-Based Retrieval (LlamaIndex)
# ============================================================================

def build_retriever(
    docs_repo_path: Path,
    similarity_top_k: int = 10,
    verbose: bool = True
):
    """
    Build a reusable retriever from documentation repository.

    This function builds the vector index ONCE and returns a retriever object
    that can be reused for multiple queries. This is much more efficient than
    rebuilding the index for every retrieval call.

    **Use Case:**
    When you need to retrieve documentation for multiple tasks from the same
    repository, build the retriever once and reuse it:

    ```python
    # Build once
    retriever = build_retriever(docs_repo_path, similarity_top_k=10)

    # Reuse for multiple tasks
    for task in tasks:
        docs = retrieve_with_retriever(retriever, task.description, top_k=5)
        # ... use docs ...
    ```

    **Performance:**
    - Building the index is expensive (embedding creation)
    - First call: ~10-30 seconds for large doc repositories
    - Subsequent queries: <1 second each

    Args:
        docs_repo_path: Path to NetworkX documentation JSON file
        similarity_top_k: Maximum number of documents to retrieve per query
            Set this higher than what you'll use in individual queries
            (default: 10, recommended: 10-20)
        verbose: Whether to print build progress (default: True)

    Returns:
        Retriever object that can be passed to retrieve_with_retriever()
        Returns None if LlamaIndex not available or build fails

    Raises:
        ImportError: If LlamaIndex not installed (caught internally, returns None)
        FileNotFoundError: If documentation repository doesn't exist

    Example:
        >>> retriever = build_retriever(
        ...     Path("networkx_docs.json"),
        ...     similarity_top_k=15
        ... )
        >>> # Now use retriever for multiple queries
        >>> docs1 = retrieve_with_retriever(retriever, "shortest path", top_k=5)
        >>> docs2 = retrieve_with_retriever(retriever, "clustering", top_k=3)

    Note:
        - The retriever is stateful - keep the reference as long as you need it
        - Building once and reusing is 10-100x faster than repeated builds
        - The retriever uses similarity_top_k as the maximum candidates to consider
    """
    # Check if LlamaIndex is available
    if not LLAMA_INDEX_AVAILABLE:
        if verbose:
            print("⚠ Warning: LlamaIndex not available, cannot build retriever")
            print("  Install with: pip install llama-index")
        return None

    try:
        # Step 1: Load and flatten the documentation repository
        if verbose:
            print(f"[BuildRetriever] Loading documentation from {docs_repo_path.name}...")

        repo = load_repository(str(docs_repo_path))
        docs = flatten_repo(repo)

        if not docs:
            if verbose:
                print("⚠ Warning: No documents found in repository")
            return None

        if verbose:
            print(f"[BuildRetriever] Extracted {len(docs)} function documentations")

        # Step 2: Create LlamaIndex Document objects
        func_names, docstrings = zip(*docs)
        documents = [Document(text=docstring) for docstring in docstrings]

        # Step 3: Build vector index with embeddings (THIS IS THE EXPENSIVE PART)
        if verbose:
            print(f"[BuildRetriever] Building vector index (creating embeddings)...")
            print(f"                This may take 10-30 seconds...")
            _estimate_embedding_cost(
                [d.text for d in documents],
                label=f"Embedding {len(documents)} documents (build_retriever)",
                verbose=True
            )

        index = VectorStoreIndex.from_documents(documents)

        # Step 4: Create retriever
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)

        if verbose:
            print(f"[BuildRetriever] ✓ Retriever built successfully!")
            print(f"                  Ready for {len(docs)} documents with top-k={similarity_top_k}")
            print(f"                  Each subsequent query costs ~${EMBEDDING_COST_PER_1K_TOKENS * 0.05:.6f} (1 query embedding)")

        return retriever

    except FileNotFoundError as e:
        if verbose:
            print(f"⚠ Error: Documentation file not found: {docs_repo_path}")
        raise
    except Exception as e:
        if verbose:
            print(f"⚠ Error building retriever: {e}")
            import traceback
            traceback.print_exc()
        return None


def retrieve_with_retriever(
    retriever,
    task_description: str,
    top_k: int = 5,
    similarity_threshold: float = 0.70,
    verbose: bool = True
) -> List[str]:
    """
    Retrieve documents using a pre-built retriever.

    This function performs fast retrieval using a pre-built retriever object.
    It does NOT rebuild the vector index, making it much faster than
    retrieve_docs_with_vector_store() for multiple queries.

    **Performance:**
    - No index building overhead
    - Typical query time: <1 second
    - 10-100x faster than rebuilding index each time

    Args:
        retriever: Pre-built retriever from build_retriever()
        task_description: Natural language description of the task
        top_k: Number of documents to retrieve (default: 5)
            Must be <= similarity_top_k used when building retriever
        similarity_threshold: Minimum similarity score (default: 0.70)
            Range: [0.0, 1.0] where 1.0 is perfect match
        verbose: Whether to print retrieval statistics (default: True)

    Returns:
        List of relevant documentation strings filtered by threshold
        Ordered by similarity score (highest first)
        Empty list if retrieval fails or no docs above threshold

    Example:
        >>> retriever = build_retriever(Path("networkx_docs.json"))
        >>> docs = retrieve_with_retriever(
        ...     retriever,
        ...     "Find shortest path",
        ...     top_k=5,
        ...     similarity_threshold=0.70
        ... )
        >>> print(f"Retrieved {len(docs)} documents")

    Note:
        - Retriever must be built with build_retriever() first
        - If retriever is None, returns empty list
        - This function is designed for high-throughput batch retrieval
    """
    # Handle None retriever (build failed or LlamaIndex unavailable)
    if retriever is None:
        if verbose:
            print("⚠ Warning: Retriever is None, cannot perform retrieval")
        return []

    try:
        # Perform retrieval with pre-built index
        if verbose:
            print(f"  [Retrieval] Retrieving top-{top_k} similar documents...")
            query_cost = _estimate_embedding_cost(
                [task_description],
                label="Query embedding (1 query)",
                verbose=True
            )

        results = retriever.retrieve(task_description)

        if not results:
            if verbose:
                print("⚠ Warning: No results returned from retrieval")
            return []

        # Filter by similarity threshold
        filtered_results = [r.text for r in results[:top_k] if r.score >= similarity_threshold]

        # Display retrieval statistics
        if verbose:
            max_similarity = max([r.score for r in results[:top_k]]) if results else 0.0
            avg_similarity = sum([r.score for r in results[:top_k]]) / min(len(results), top_k) if results else 0.0

            print(f"  [Retrieval] Statistics:")
            print(f"              - Max similarity score: {max_similarity:.3f}")
            print(f"              - Avg similarity score: {avg_similarity:.3f}")
            print(f"              - Docs above threshold ({similarity_threshold}): {len(filtered_results)}/{min(len(results), top_k)}")

            # Show preview of best match
            if filtered_results:
                preview = filtered_results[0][:150] + "..." if len(filtered_results[0]) > 150 else filtered_results[0]
                print(f"              - Best match preview: {preview}")

        return filtered_results

    except Exception as e:
        if verbose:
            print(f"⚠ Warning: Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
        return []


def retrieve_docs_with_vector_store(
    docs_repo_path: Path,
    task_description: str,
    top_k: int = 5,
    similarity_threshold: float = 0.70,
    verbose: bool = True
) -> List[str]:
    """
    Retrieve relevant documentation using LlamaIndex vector store.

    This implements vector-based retrieval using LlamaIndex embeddings:

    **Process:**
    1. Load NetworkX documentation from JSON file
    2. Flatten nested structure to extract all docstrings
    3. Create LlamaIndex Document objects for each docstring
    4. Build VectorStoreIndex with embeddings for all documents
    5. Use retriever to find top-k most similar documents to query
    6. Filter results by similarity threshold to ensure quality
    7. Return filtered documentation strings

    **Technical Details:**
    - Uses LlamaIndex VectorStoreIndex for embedding-based retrieval
    - Default embedding model: OpenAI embeddings (via LlamaIndex)
    - Alternative: sentence-transformers if OpenAI unavailable
    - Similarity metric: Cosine similarity between query and doc embeddings
    - Threshold filtering ensures only high-quality matches are returned

    Args:
        docs_repo_path: Path to NetworkX documentation JSON file
            Expected: crawl_documentation/networkx_graph_functions_docs.json
        task_description: Natural language description of the graph task
            Example: "Find the shortest path between two nodes in a weighted graph"
        top_k: Number of top documents to retrieve before filtering (default: 5)
            Higher values consider more candidates but may include less relevant docs
        similarity_threshold: Minimum similarity score to include (default: 0.70)
            Range: [0.0, 1.0] where 1.0 is perfect match
            Recommended: 0.60-0.80 for balanced precision/recall
        verbose: Whether to print retrieval statistics (default: True)

    Returns:
        List of relevant documentation strings filtered by threshold
        Ordered by similarity score (highest first)
        Empty list if:
        - Retrieval fails
        - No docs above threshold
        - LlamaIndex not available

    Raises:
        ImportError: If LlamaIndex not installed (caught internally, returns [])
        FileNotFoundError: If documentation repository doesn't exist (caught internally)

    Example:
        >>> docs = retrieve_docs_with_vector_store(
        ...     Path("networkx_docs.json"),
        ...     "Find shortest path between two nodes",
        ...     top_k=5,
        ...     similarity_threshold=0.70
        ... )
        >>> print(f"Retrieved {len(docs)} relevant documents")
        Retrieved 3 relevant documents

    Note:
        - First call may be slow due to embedding creation
        - Subsequent calls on same data are faster (cached embeddings)
        - Requires llama-index package: pip install llama-index
    """
    # Check if LlamaIndex is available
    if not LLAMA_INDEX_AVAILABLE:
        if verbose:
            print("⚠ Warning: LlamaIndex not available, skipping retrieval")
            print("  Install with: pip install llama-index")
        return []

    try:
        # Step 1: Load and flatten the documentation repository
        if verbose:
            print(f"  [Retrieval] Loading documentation from {docs_repo_path.name}...")

        repo = load_repository(str(docs_repo_path))
        docs = flatten_repo(repo)

        if not docs:
            if verbose:
                print("⚠ Warning: No documents found in repository")
            return []

        if verbose:
            print(f"  [Retrieval] Extracted {len(docs)} function documentations")

        # Step 2: Create LlamaIndex Document objects
        func_names, docstrings = zip(*docs)
        documents = [Document(text=docstring) for docstring in docstrings]

        # Step 3: Build vector index with embeddings
        if verbose:
            print(f"  [Retrieval] Building vector index (creating embeddings)...")
            print(f"              This may take a moment on first run...")
            doc_cost = _estimate_embedding_cost(
                [d.text for d in documents],
                label=f"Embedding {len(documents)} documents (index build)",
                verbose=True
            )
            query_cost = _estimate_embedding_cost(
                [task_description],
                label=f"Embedding 1 query",
                verbose=True
            )
            print(f"  [API Cost] Total estimated cost for this call: ${doc_cost + query_cost:.6f}")

        index = VectorStoreIndex.from_documents(documents)

        # Step 4: Create retriever and retrieve top-k similar documents
        retriever = index.as_retriever(similarity_top_k=top_k)

        if verbose:
            print(f"  [Retrieval] Retrieving top-{top_k} similar documents...")

        results = retriever.retrieve(task_description)

        if not results:
            if verbose:
                print("⚠ Warning: No results returned from retrieval")
            return []

        # Step 5: Filter by similarity threshold
        filtered_results = [r.text for r in results if r.score >= similarity_threshold]

        # Step 6: Display retrieval statistics
        if verbose:
            max_similarity = max([r.score for r in results]) if results else 0.0
            avg_similarity = sum([r.score for r in results]) / len(results) if results else 0.0

            print(f"  [Retrieval] Statistics:")
            print(f"              - Max similarity score: {max_similarity:.3f}")
            print(f"              - Avg similarity score: {avg_similarity:.3f}")
            print(f"              - Docs above threshold ({similarity_threshold}): {len(filtered_results)}/{len(results)}")

            # Show preview of best match
            if filtered_results:
                preview = filtered_results[0][:150] + "..." if len(filtered_results[0]) > 150 else filtered_results[0]
                print(f"              - Best match preview: {preview}")

        return filtered_results

    except FileNotFoundError as e:
        if verbose:
            print(f"⚠ Warning: {e}")
        return []
    except Exception as e:
        if verbose:
            print(f"⚠ Warning: Error during retrieval: {e}")
            # Print traceback for debugging
            import traceback
            print("  Full error:")
            traceback.print_exc()
        return []


# ============================================================================
# Pre-Retrieved Documentation (Cached)
# ============================================================================

def load_pre_retrieved_docs(
    docs_file_path: Optional[Path],
    task_name: str,
    verbose: bool = False
) -> List[str]:
    """
    Load pre-retrieved documentation from a cached file.

    This is an alternative to on-the-fly retrieval for faster execution.
    If you've already run retrieval once, you can save the results and
    load them directly without rebuilding the vector index each time.

    Args:
        docs_file_path: Path to JSON file containing pre-retrieved docs
            Expected format: {"task_name": ["doc1", "doc2", ...], ...}
        task_name: Name of the task to get docs for
        verbose: Whether to print loading messages (default: False)

    Returns:
        List of documentation strings for this task
        Empty list if file doesn't exist or task not found

    Example:
        >>> docs = load_pre_retrieved_docs(
        ...     Path("cached_docs.json"),
        ...     "clustering"
        ... )
        >>> len(docs)
        5

    Note:
        - Much faster than on-the-fly retrieval
        - Requires pre-retrieval step to generate cache file
        - Cache file can be shared across experiments
    """
    if docs_file_path is None or not docs_file_path.exists():
        if verbose:
            print(f"⚠ Warning: Cached docs file not found at {docs_file_path}")
        return []

    try:
        with open(docs_file_path, 'r', encoding='utf-8') as f:
            all_docs = json.load(f)

        # Get docs for this specific task
        task_docs = all_docs.get(task_name, [])

        if not task_docs:
            if verbose:
                print(f"⚠ Warning: No cached docs found for task '{task_name}'")
            return []

        return task_docs if isinstance(task_docs, list) else [task_docs]

    except (json.JSONDecodeError, Exception) as e:
        if verbose:
            print(f"⚠ Warning: Error loading cached docs: {e}")
        return []


# ============================================================================
# Convenience Function
# ============================================================================

def retrieve_or_load_docs(
    task_name: str,
    task_description: str,
    docs_repo_path: Optional[Path] = None,
    docs_file_path: Optional[Path] = None,
    top_k: int = 5,
    similarity_threshold: float = 0.70,
    verbose: bool = True
) -> Tuple[List[str], str]:
    """
    Retrieve or load documentation with automatic fallback.

    This convenience function tries multiple retrieval strategies in order:
    1. Load from cached file (fastest)
    2. Retrieve on-the-fly with LlamaIndex (if docs_repo provided)
    3. Return empty list (zero-shot)

    Args:
        task_name: Name of the task (for cache lookup)
        task_description: Natural language task description (for retrieval)
        docs_repo_path: Path to documentation JSON for on-the-fly retrieval
        docs_file_path: Path to cached documentation JSON
        top_k: Number of docs to retrieve (default: 5)
        similarity_threshold: Minimum similarity score (default: 0.70)
        verbose: Whether to print progress messages (default: True)

    Returns:
        Tuple of (documentation_list, retrieval_method)
        where retrieval_method is one of: "cached", "vector", "zero-shot"

    Example:
        >>> docs, method = retrieve_or_load_docs(
        ...     task_name="clustering",
        ...     task_description="Find clustering coefficient",
        ...     docs_repo_path=Path("nx_docs.json"),
        ...     docs_file_path=Path("cached.json")
        ... )
        >>> print(f"Used {method}: {len(docs)} docs")
        Used cached: 3 docs
    """
    retrieved_docs = []
    method = "zero-shot"

    # Strategy 1: Load from cache (fastest)
    if docs_file_path and docs_file_path.exists():
        if verbose:
            print(f"  [Mode] Loading from pre-retrieved cache: {docs_file_path.name}")

        retrieved_docs = load_pre_retrieved_docs(docs_file_path, task_name, verbose=verbose)

        if retrieved_docs:
            method = "cached"
            if verbose:
                print(f"✓ Loaded {len(retrieved_docs)} cached documentation snippets")
        else:
            if verbose:
                print(f"⚠ No cached docs found for task '{task_name}'")

    # Strategy 2: Retrieve on-the-fly with embeddings
    if not retrieved_docs and docs_repo_path and docs_repo_path.exists():
        if verbose:
            print(f"  [Mode] Retrieving on-the-fly with vector embeddings")

        retrieved_docs = retrieve_docs_with_vector_store(
            docs_repo_path=docs_repo_path,
            task_description=task_description,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            verbose=verbose
        )

        if retrieved_docs:
            method = "vector"
            if verbose:
                print(f"✓ Retrieved {len(retrieved_docs)} documentation snippets\n")
        else:
            if verbose:
                print(f"⚠ Retrieval failed or no docs above threshold\n")

    # Strategy 3: Zero-shot (no docs)
    if not retrieved_docs:
        if verbose:
            print(f"⚠ No documentation available, using zero-shot approach\n")
        method = "zero-shot"

    return retrieved_docs, method


# ============================================================================
# Utility Functions
# ============================================================================

def is_llamaindex_available() -> bool:
    """
    Check if LlamaIndex is available for import.

    Returns:
        True if LlamaIndex can be imported, False otherwise

    Example:
        >>> if is_llamaindex_available():
        ...     print("Can use vector retrieval")
        ... else:
        ...     print("Use cached docs or zero-shot")
    """
    return LLAMA_INDEX_AVAILABLE


def save_retrieved_docs_to_cache(
    retrieved_docs: Dict[str, List[str]],
    output_path: Path
) -> None:
    """
    Save retrieved documentation to a cache file for reuse.

    Args:
        retrieved_docs: Dictionary mapping task names to doc lists
            Format: {"task_name": ["doc1", "doc2", ...], ...}
        output_path: Path where to save the cache file

    Example:
        >>> docs_cache = {
        ...     "clustering": ["nx.clustering(G, node)...", ...],
        ...     "diameter": ["nx.diameter(G)...", ...]
        ... }
        >>> save_retrieved_docs_to_cache(docs_cache, Path("cached_docs.json"))
    """
    # Ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON with proper formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(retrieved_docs, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved retrieved docs cache to: {output_path}")


# ============================================================================
# Example Usage (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the LlamaIndex retrieval functionality.

    To run this example:
        python -m utils.llamaindex_retrieval
    """
    import sys

    print("LlamaIndex Retrieval Utilities")
    print("="*60)
    print(f"LlamaIndex available: {is_llamaindex_available()}")
    print()

    if len(sys.argv) < 2:
        print("Usage: python -m utils.llamaindex_retrieval <path_to_docs.json>")
        print("\nExample:")
        print("  python -m utils.llamaindex_retrieval crawl_documentation/networkx_graph_functions_docs.json")
        sys.exit(1)

    repo_path = Path(sys.argv[1])

    # Example query
    query = "How to find the shortest path between two nodes in a graph?"
    print(f"Query: {query}\n")

    # Test retrieval
    if is_llamaindex_available():
        print("Testing vector-based retrieval:")
        print("-"*60)
        docs = retrieve_docs_with_vector_store(
            docs_repo_path=repo_path,
            task_description=query,
            top_k=3,
            similarity_threshold=0.70,
            verbose=True
        )

        print(f"\nRetrieved {len(docs)} documents")
        if docs:
            print("\nFirst document preview:")
            print(docs[0][:300] + "...")
    else:
        print("⚠ LlamaIndex not available")
        print("  Install with: pip install llama-index")
