"""
Retrieval Agent for NetworkX Documentation

This module provides an intelligent agentic retrieval system that navigates
hierarchical documentation structures using LLM-guided decisions. Unlike
simple keyword matching (TF-IDF) or semantic similarity (Sentence-BERT),
this agent actively explores the documentation tree by asking an LLM to
choose the most relevant categories at each level.

Documentation Structure:
    The JSON documentation repository has a nested structure:

    {
        "category_name": [
            "category_description",           # Index 0: Description of this category
            {                                  # Index 1: Nested content (dict or string)
                "subcategory": [...],          # Recursively nested
                "function_name": [
                    "function_summary",        # Index 0: Brief summary
                    "full_docstring"           # Index 1: Complete documentation
                ]
            }
        ]
    }

    Each entry is a 2-element list:
    - [0]: Description/summary text
    - [1]: Either a dict (more nesting) or a string (leaf docstring)

Algorithm:
    1. Start at root level of documentation tree
    2. Ask LLM to select most relevant categories from current level
    3. Navigate into selected categories
    4. Repeat until reaching leaf nodes (docstrings)
    5. Ask LLM to choose best docstring among collected leaves
    6. If no valid docstring found, backtrack and try different path
    7. Repeat up to max_rounds times with different initial choices

Key Features:
- Multi-round exploration with backtracking on failure
- LLM-guided navigation through nested documentation menus
- Handles arbitrary depth nested JSON structures
- Configurable exploration depth and rounds
- Text normalization for Unicode and whitespace handling

Example:
    from utils.retrieval_agent import retrieve_doc
    from utils.llm_agent.openai_code_generator import OpenAICodeGenerator

    llm = OpenAICodeGenerator(model="gpt-4")
    docs = retrieve_doc(
        doc_path="docs/networkx_docs.json",
        user_query="How to find shortest path?",
        llm_model=llm,
        max_rounds=3,
        top_k=3
    )
"""

import json
import re
import unicodedata
from typing import List, Tuple, Optional, Union

from utils.llm_agent.deepseek_code_generator import DeepSeekCodeGenerator
from utils.llm_agent.openai_code_generator import OpenAICodeGenerator
from utils.generation_functions.retrieve_doc_chapter import retrieve_documentation_chapter
from utils.generation_functions.get_most_relevant_doc import get_most_relevant_doc
from utils.sentbert_retrieval import find_best_docstring


# ============================================================================
# DeepSeek-Chat API Pricing (used as retrieval agent)
# ============================================================================
# Pricing: https://platform.deepseek.com/api-docs/pricing
DEEPSEEK_INPUT_COST_PER_1M_TOKENS = 0.028   # $0.028 / 1M input tokens (cache hit)
DEEPSEEK_OUTPUT_COST_PER_1M_TOKENS = 0.42    # $0.42 / 1M output tokens


def _estimate_retrieval_cost(input_text: str, output_tokens_est: int = 100, label: str = "", cost_tracker: list = None) -> dict:
    """
    Estimate the DeepSeek API cost for a retrieval agent LLM call.

    Args:
        input_text: The full input text (prompt) sent to the LLM.
        output_tokens_est: Estimated output tokens (default: 100, typical for selection tasks).
        label: Description of this call for logging.
        cost_tracker: Optional list to accumulate cost dicts for total calculation.

    Returns:
        Dict with input_tokens, output_tokens, input_cost, output_cost, total_cost.
    """
    input_tokens = len(input_text) // 4  # ~4 chars per token
    input_cost = (input_tokens / 1_000_000) * DEEPSEEK_INPUT_COST_PER_1M_TOKENS
    output_cost = (output_tokens_est / 1_000_000) * DEEPSEEK_OUTPUT_COST_PER_1M_TOKENS
    total_cost = input_cost + output_cost

    print(f"  [Retrieval Cost] {label}")
    print(f"    Input: ~{input_tokens:,} tokens (${input_cost:.6f}), Output: ~{output_tokens_est:,} tokens (${output_cost:.6f})")
    print(f"    Total: ${total_cost:.6f} (deepseek-chat, input@${DEEPSEEK_INPUT_COST_PER_1M_TOKENS}/1M, output@${DEEPSEEK_OUTPUT_COST_PER_1M_TOKENS}/1M)")

    result = {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens_est,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost
    }

    if cost_tracker is not None:
        cost_tracker.append(result)

    return result


def _print_total_cost(cost_tracker: list, num_rounds: int):
    """Print the total accumulated retrieval cost from all LLM calls."""
    if not cost_tracker:
        print(f"\n  [Total Retrieval Cost] No LLM calls made.")
        return
    total_input_tokens = sum(c['input_tokens'] for c in cost_tracker)
    total_output_tokens = sum(c['output_tokens'] for c in cost_tracker)
    total_input_cost = sum(c['input_cost'] for c in cost_tracker)
    total_output_cost = sum(c['output_cost'] for c in cost_tracker)
    total_cost = sum(c['total_cost'] for c in cost_tracker)
    print(f"\n  [Total Retrieval Cost] {len(cost_tracker)} LLM calls across {num_rounds} round(s)")
    print(f"    Total input:  ~{total_input_tokens:,} tokens (${total_input_cost:.6f})")
    print(f"    Total output: ~{total_output_tokens:,} tokens (${total_output_cost:.6f})")
    print(f"    Total cost:   ${total_cost:.6f}")


# ============================================================================
# Text Normalization Constants and Functions
# ============================================================================

# Regular expression patterns for cleaning text
ZERO_WIDTH_CHARS = r'[\u200B-\u200D\uFEFF]'  # Zero-width Unicode characters (invisible)
WHITESPACE_PATTERN = r'\s+'                   # All Unicode whitespace characters


def normalize_text(text: str) -> str:
    """
    Normalize text by removing invisible characters and standardizing whitespace.

    This function performs several text cleaning operations:
    1. Normalize Unicode to NFC form (canonical composition)
    2. Remove zero-width invisible characters
    3. Collapse all whitespace sequences into single spaces
    4. Strip leading/trailing whitespace

    Args:
        text: Input text to normalize

    Returns:
        Cleaned and normalized text

    Example:
        >>> normalize_text("hello  \\u200B world\\n\\ttab")
        'hello world tab'
    """
    # Step 1: Normalize Unicode to canonical form
    # NFC = Canonical Composition (e.g., é as single character, not e + accent)
    text = unicodedata.normalize('NFC', text)

    # Step 2: Remove zero-width invisible characters
    # These are characters that don't display but can cause string comparison issues
    text = re.sub(ZERO_WIDTH_CHARS, '', text)

    # Step 3: Collapse all whitespace (spaces, tabs, newlines) into single spaces
    text = re.sub(WHITESPACE_PATTERN, ' ', text, flags=re.UNICODE)

    # Step 4: Remove leading and trailing whitespace
    return text.strip()


# ============================================================================
# Documentation Repository Navigation
# ============================================================================

def load_documentation_repo(doc_path: str) -> dict:
    """
    Load the documentation repository from a JSON file.

    Args:
        doc_path: Path to the JSON documentation file

    Returns:
        Dictionary containing the hierarchical documentation structure

    Raises:
        FileNotFoundError: If the documentation file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(doc_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_categories_and_descriptions(documentation: dict) -> Tuple[List[str], dict]:
    """
    Extract top-level categories and their descriptions from documentation.

    The documentation structure is:
        {
            "category_name": ["description", {...nested content...}],
            ...
        }

    Args:
        documentation: Root-level documentation dictionary

    Returns:
        Tuple of:
        - List of category names (e.g., ["Algorithms", "Graph Classes", "Functions"])
        - Dictionary mapping category names to descriptions

    Example:
        >>> doc = {"Algorithms": ["Graph algorithms", {...}], "Utils": ["Utilities", {...}]}
        >>> categories, descriptions = extract_categories_and_descriptions(doc)
        >>> print(categories)
        ['Algorithms', 'Utils']
        >>> print(descriptions)
        {'Algorithms': 'Graph algorithms', 'Utils': 'Utilities'}
    """
    category_names = list(documentation.keys())

    # Each category's value is a list [description, nested_content]
    # Extract the description (index 0) for each category
    descriptions = [documentation[key][0] for key in category_names]

    # Create a mapping of category name to description
    category_to_description = dict(zip(category_names, descriptions))

    return category_names, category_to_description


def get_nested_content(documentation: dict, category: str):
    """
    Get the nested content for a given category.

    In the documentation structure, each entry is:
        "category_name": ["description", nested_content]

    This function returns nested_content (index 1), which can be:
    - A dictionary (more nested categories)
    - A string (leaf node - the actual docstring)

    Args:
        documentation: Documentation dictionary
        category: Category name to retrieve

    Returns:
        The nested content (dict or str) at index 1

    Raises:
        KeyError: If category doesn't exist
        IndexError: If the category value is not a list with at least 2 elements
    """
    return documentation[category][1]


def collect_children_from_current_level(
    current_menus: List[dict],
    current_choices: List[List[str]]
) -> List[Union[dict, str]]:
    """
    Collect all child nodes from the current level based on LLM's choices.

    This function takes the current menu dictionaries and the LLM's category choices,
    and extracts all the nested content for those choices.

    Args:
        current_menus: List of menu dictionaries at the current level
        current_choices: List of category choices for each menu (parallel to current_menus)
                        Each element is a list of category names chosen by the LLM

    Returns:
        List of children (can be dicts for deeper nesting or strings for docstrings)
        Duplicates are removed to avoid processing the same content multiple times

    Example:
        >>> menus = [{"cat1": ["desc", "docstring"], "cat2": ["desc", {...nested...}]}]
        >>> choices = [["cat1", "cat2"]]
        >>> children = collect_children_from_current_level(menus, choices)
        >>> # Returns: ["docstring", {...nested...}]

    Raises:
        KeyError: If a chosen category doesn't exist in the menu
        ValueError: If the menu structure is invalid
    """
    print("### current choices:", current_choices)
    children = []

    # Iterate through each menu and its corresponding choices
    # menu 的意思就是当前的目录节点
    for menu_idx, menu in enumerate(current_menus):
        # Skip if this menu has no choices
        if current_choices[menu_idx] is None:
            continue

        # For each category chosen by the LLM
        for category_name in current_choices[menu_idx]:
            # Skip None values (can occur from filtering)
            if category_name is None:
                continue

            try:
                # Extract the nested content (index 1) for this category
                child = get_nested_content(menu, category_name)

                # Avoid duplicates (same content can appear in multiple paths)
                if child not in children:
                    children.append(child)

            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid menu structure or unknown category '{category_name}' during traversal"
                ) from exc
    print("### children.keys:", [child.keys() if isinstance(child, dict) else "[content]" for child in children])
    return children


def process_children_for_next_level(
    children: List[Union[dict, str]],
    user_query: str,
    llm_model,
    top_k: int,
    llm_decision: bool,
    collected_docstrings: List[Optional[str]],
    cost_tracker: list = None
) -> Tuple[List[dict], List[List[str]]]:
    """
    Process children to prepare for the next level of navigation.

    Children can be:
    1. Dictionaries (categories with more nesting) - Ask LLM to select subcategories
    2. Strings (leaf docstrings) - Collect them for final selection

    Args:
        children: List of child nodes (dicts or strings) from current level
        user_query: User's search query
        llm_model: LLM model for making category selections
        top_k: Maximum number of categories to consider
        llm_decision: Whether to use LLM for decision making
        collected_docstrings: List to accumulate found docstrings (modified in place)

    Returns:
        Tuple of:
        - next_menus: List of dict menus for the next level
        - next_choices: List of category choices for each next menu

    Example:
        If children = [
            {"subcategory1": [...], "subcategory2": [...]},  # Dict - more nesting
            "This is a docstring"                            # String - leaf node
        ]

        Then:
        - The dict will be added to next_menus
        - The LLM will choose which subcategories to explore
        - The string will be added to collected_docstrings
    """
    next_menus = []
    next_choices = []

    for child in children:
        # Case 1: Dictionary - More nested categories to explore
        if isinstance(child, dict):
            subcategory_names = list(child.keys())

            # If there are many subcategories (> top_k), ask LLM to filter
            if len(subcategory_names) > top_k:
                # Extract descriptions for each subcategory
                subcategory_descriptions = [child[name][0] for name in subcategory_names]
                category_to_desc = dict(zip(subcategory_names, subcategory_descriptions))

                # Ask LLM: "Which of these subcategories are most relevant to the query?"
                _estimate_retrieval_cost(
                    input_text=user_query + str(category_to_desc),
                    output_tokens_est=50,
                    label=f"Subcategory selection ({len(subcategory_names)} subcategories)",
                    cost_tracker=cost_tracker
                )
                selected_categories = retrieve_documentation_chapter(
                    query=user_query,
                    openai_model=llm_model,
                    chapters_and_descs=category_to_desc,
                    llm_descision=llm_decision
                )
            else:
                # If few subcategories, explore all of them
                selected_categories = subcategory_names

            # Add this menu and choices to the next level
            next_menus.append(child)
            next_choices.append(selected_categories)

        # Case 2: String - Leaf node (docstring)
        elif isinstance(child, str):
            # Collect this docstring for final LLM selection
            collected_docstrings.append(child)

        # Case 3: Other types (rare) - Record as None
        else:
            collected_docstrings.append(None)

    return next_menus, next_choices


def select_best_docstring_from_collected(
    user_query: str,
    llm_model,
    collected_docstrings: List[Optional[str]],
    composite_task: bool,
    cost_tracker: list = None
) -> Optional[Union[str, List[str]]]:
    """
    Select the most relevant docstring(s) from all collected leaf nodes.

    After traversing the documentation tree and collecting multiple docstrings,
    this function asks the LLM to choose which one(s) are most relevant to the query.

    Args:
        user_query: User's search query
        llm_model: LLM model for making the selection
        collected_docstrings: List of docstrings collected during traversal
        composite_task: Whether this is a composite task (may return multiple docs)

    Returns:
        - Single docstring (str) for simple tasks
        - List of docstrings (List[str]) for composite tasks
        - None if no valid docstring found or selection fails

    Example:
        >>> docstrings = ["Doc about shortest path", "Doc about longest path", "Doc about BFS"]
        >>> result = select_best_docstring_from_collected(
        ...     "How to find shortest path?", llm, docstrings, False
        ... )
        >>> print(result)
        "Doc about shortest path"
    """
    # Ask LLM to analyze all docstrings and pick the most relevant one(s)
    total_doc_text = " ".join(str(d) for d in collected_docstrings if d)
    _estimate_retrieval_cost(
        input_text=user_query + total_doc_text,
        output_tokens_est=200,
        label=f"Final doc selection ({len([d for d in collected_docstrings if d])} candidate docstrings)",
        cost_tracker=cost_tracker
    )
    selected_docs = get_most_relevant_doc(
        user_query,
        llm_model,
        collected_docstrings,
        composite_task=composite_task
    )

    # Handle the return value
    if selected_docs is None:
        return None

    # Ensure we return a list for composite tasks, single doc otherwise
    if composite_task:
        return selected_docs if isinstance(selected_docs, list) else [selected_docs]
    else:
        return selected_docs if isinstance(selected_docs, list) else [selected_docs]


# ============================================================================
# Single Round Traversal
# ============================================================================

def traverse_documentation_one_round(
    doc_path: str,
    user_query: str,
    llm_model,
    explored_initial_categories: List[str],
    max_depth: int,
    top_k: int,
    llm_decision: bool,
    composite_task: bool,
    cost_tracker: list = None
) -> Tuple[List[str], Optional[Union[str, List[str]]]]:
    """
    Perform one complete traversal of the documentation tree.

    This function implements a single exploration attempt:
    1. Load documentation and get top-level categories
    2. Ask LLM to select initial categories (avoiding previously explored ones)
    3. Navigate level by level, asking LLM to choose subcategories
    4. Collect all reached leaf docstrings
    5. Ask LLM to select the most relevant docstring(s)

    Args:
        doc_path: Path to documentation JSON file
        user_query: User's search query
        llm_model: LLM model for navigation decisions
        explored_initial_categories: Categories already tried in previous rounds (to avoid)
        max_depth: Maximum depth to traverse (safety limit)
        top_k: Number of categories to consider at each level
        llm_decision: Whether to use LLM for decisions
        composite_task: Whether this is a composite task

    Returns:
        Tuple of:
        - List of initial categories chosen in this round
        - Selected docstring(s) or None if traversal failed

    Example:
        >>> first_choices, docs = traverse_documentation_one_round(
        ...     "docs.json", "shortest path", llm, [], 5, 3, True, False
        ... )
        >>> print(first_choices)
        ['Algorithms', 'Path Finding']
        >>> print(docs)
        "Documentation about shortest path algorithms..."
    """
    # ========================================================================
    # Step 1: Load documentation and extract top-level structure
    # ========================================================================

    documentation = load_documentation_repo(doc_path)

    # Get all top-level categories (e.g., "Algorithms", "Graph Classes", etc.)
    category_names, category_descriptions = extract_categories_and_descriptions(documentation)

    # ========================================================================
    # Step 2: Ask LLM to select initial categories to explore
    # ========================================================================

    if len(explored_initial_categories) == 0:
        # First round: Ask LLM to choose most relevant categories
        _estimate_retrieval_cost(
            input_text=user_query + str(category_descriptions),
            output_tokens_est=50,
            label=f"Initial category selection ({len(category_names)} categories)",
            cost_tracker=cost_tracker
        )
        initial_category_choices = retrieve_documentation_chapter(
            query=user_query,
            openai_model=llm_model,
            chapters_and_descs=category_descriptions,
            llm_descision=llm_decision
        )
    else:
        # Subsequent rounds: Ask LLM to choose different categories
        # Pass explored_choices to encourage trying alternative paths
        _estimate_retrieval_cost(
            input_text=user_query + str(category_descriptions) + str(explored_initial_categories),
            output_tokens_est=50,
            label=f"Initial category selection round {len(explored_initial_categories)//top_k + 1} ({len(category_names)} categories)",
            cost_tracker=cost_tracker
        )
        initial_category_choices = retrieve_documentation_chapter(
            query=user_query,
            openai_model=llm_model,
            chapters_and_descs=category_descriptions,
            explored_choices=explored_initial_categories,
            llm_descision=llm_decision
        )

    # Record which top-level categories were chosen (for backtracking)
    first_level_categories = initial_category_choices

    # ========================================================================
    # Step 3: Initialize traversal state
    # ========================================================================

    # Normalize category names to handle Unicode/whitespace variations
    normalized_choices = [
        [normalize_text(choice.strip()) for choice in initial_category_choices]
    ]

    # State tracking:
    # - current_menus: List of dictionaries representing current menu level
    # - current_choices: List of category names to explore at this level
    # - collected_docstrings: Accumulated leaf docstrings during traversal
    current_menus = [documentation]
    current_choices = normalized_choices
    collected_docstrings: List[Optional[str]] = []
    current_depth = 0

    # ========================================================================
    # Step 4: Traverse documentation tree level by level
    # ========================================================================

    while True:
        # Safety check: Prevent infinite loops
        if max_depth is not None and current_depth >= max_depth:
            raise RecursionError(
                f"Maximum traversal depth ({max_depth}) exceeded. "
                "This may indicate a circular reference in the documentation structure."
            )

        # Base case: Check if we've finished exploring all paths
        # Filter out None values and check if any choices remain
        print('Current Depth:', current_depth)
        remaining_choices = [choice for choice in current_choices if choice is not None]
        print("### remaining choices:", remaining_choices)
        if len(remaining_choices) == 0:
            # No more categories to explore - we've reached leaf level

            print("### collected docstrings:", [doc[:30] for doc in collected_docstrings])
            # Ask LLM to select the best docstring from all collected
            selected_docs = select_best_docstring_from_collected(
                user_query,
                llm_model,
                collected_docstrings,
                composite_task,
                cost_tracker=cost_tracker
            )
            print("### selected docs:", [doc[:30] for doc in selected_docs] if selected_docs else selected_docs)

            # Check if we got a valid result
            if not selected_docs or selected_docs == "None":
                # Traversal failed - no valid documentation found
                return (first_level_categories or []), None

            # Success - return the selected documentation
            return (first_level_categories or []), selected_docs

        # ====================================================================
        # Step 4a: Collect children from current level
        # ====================================================================

        # Based on LLM's category choices, extract all nested content
        children = collect_children_from_current_level(current_menus, current_choices)

        # ====================================================================
        # Step 4b: Process children for next level
        # ====================================================================

        # Separate children into:
        # - Dictionaries (more nesting) -> Ask LLM to select subcategories
        # - Strings (docstrings) -> Collect for final selection
        next_menus, next_choices = process_children_for_next_level(
            children=children,
            user_query=user_query,
            llm_model=llm_model,
            top_k=top_k,
            llm_decision=llm_decision,
            collected_docstrings=collected_docstrings,
            cost_tracker=cost_tracker
        )

        # ====================================================================
        # Step 4c: Move to next level
        # ====================================================================

        current_menus = next_menus
        current_choices = next_choices
        current_depth += 1


# ============================================================================
# Main Retrieval Function with Multi-Round Exploration
# ============================================================================

def retrieve_doc(
    doc_path: str,
    user_query: str,
    llm_model,  # DeepSeekCodeGenerator | OpenAICodeGenerator
    max_depth: int = 5,
    top_k: int = 3,
    max_rounds: int = 3,
    llm_descision: bool = False,  # Note: Typo in parameter name (llm_decision) kept for compatibility
    composite_task: bool = False,
) -> Optional[Union[str, List[str]]]:
    """
    Retrieve relevant documentation using LLM-guided multi-round exploration.

    This is the main entry point for the retrieval agent. It implements a
    multi-round exploration strategy:

    1. Round 1: Start at root, ask LLM to navigate to relevant docs
    2. If Round 1 finds valid docs: Return them
    3. If Round 1 fails: Try Round 2 with different initial categories
    4. Continue up to max_rounds attempts
    5. If all rounds fail: Return None (no fallback)

    The multi-round strategy helps handle cases where the LLM's initial
    category choices lead to dead ends. By trying alternative paths,
    we increase the chance of finding relevant documentation.

    Args:
        doc_path: Path to the hierarchical documentation JSON file
        user_query: User's search query (e.g., "How to find shortest path?")
        llm_model: LLM instance for making navigation decisions
                  (DeepSeekCodeGenerator or OpenAICodeGenerator)
        _depth: Deprecated parameter (not used, kept for backward compatibility)
        max_depth: Maximum tree depth to explore (default: 5)
                  Prevents infinite loops in malformed documentation
        top_k: Number of categories to consider at each level (default: 3)
              If a menu has more than top_k items, ask LLM to filter
        max_rounds: Maximum exploration attempts (default: 3)
                   Each round tries different initial categories
        llm_descision: Whether to use LLM for decision making (default: False)
                      Note: Parameter name has typo but kept for compatibility
        composite_task: Whether this is a composite task (default: False)
                       If True, may return multiple docstrings

    Returns:
        - String: Single docstring for simple tasks
        - List[str]: Multiple docstrings for composite tasks
        - None: If all exploration rounds fail to find valid documentation

    Raises:
        RecursionError: If max_depth is exceeded during traversal
        FileNotFoundError: If doc_path doesn't exist
        json.JSONDecodeError: If documentation JSON is malformed

    Example:
        >>> from utils.llm_agent.openai_code_generator import OpenAICodeGenerator
        >>> llm = OpenAICodeGenerator(model="gpt-4")
        >>> docs = retrieve_doc(
        ...     doc_path="docs/networkx_docs.json",
        ...     user_query="How to compute shortest path between two nodes?",
        ...     llm_model=llm,
        ...     max_rounds=3,
        ...     top_k=3
        ... )
        >>> print(docs)
        "networkx.shortest_path(G, source, target)..."

    Technical Details:
        The function implements backtracking by tracking which initial categories
        have been explored. Each round tries different categories to maximize
        the chance of finding relevant documentation.

        If all rounds fail, the function returns None rather than using a fallback
        (previous versions had a Sentence-BERT fallback, now commented out).
    """
    # ========================================================================
    # Multi-Round Exploration Controller
    # ========================================================================

    print(f"\n  [Retrieval Cost Summary] DeepSeek-Chat pricing:")
    print(f"    Input:  ${DEEPSEEK_INPUT_COST_PER_1M_TOKENS}/1M tokens (cache hit)")
    print(f"    Output: ${DEEPSEEK_OUTPUT_COST_PER_1M_TOKENS}/1M tokens\n")

    # Accumulate costs from all LLM calls across all rounds
    cost_tracker = []

    # Track which top-level categories we've already explored
    # This prevents trying the same path multiple times
    explored_initial_categories = []

    # Keep track of the most recent category choices (for debugging)
    last_attempted_categories: List[str] = []

    # Try multiple exploration rounds
    for round_number in range(max_rounds):
        # Attempt one complete traversal of the documentation tree
        first_level_categories, retrieved_docs = traverse_documentation_one_round(
            doc_path=doc_path,
            user_query=user_query,
            llm_model=llm_model,
            explored_initial_categories=explored_initial_categories,
            max_depth=max_depth,
            top_k=top_k,
            llm_decision=llm_descision,  # Note: Typo in original parameter name
            composite_task=composite_task,
            cost_tracker=cost_tracker
        )

        # Record which categories were tried in this round
        last_attempted_categories = first_level_categories

        # Check if we found valid documentation
        if retrieved_docs is not None and retrieved_docs != "None":
            # Success! Return the documentation
            print(f"Retrieval succeeded in round {round_number + 1} with categories: {first_level_categories}")
            _print_total_cost(cost_tracker, round_number + 1)
            return retrieved_docs

        # This round failed - add tried categories to explored list
        # Next round will try different categories
        explored_initial_categories.extend(last_attempted_categories)

    # ========================================================================
    # All Rounds Failed
    # ========================================================================

    # After exhausting all rounds, we couldn't find valid documentation
    # Previous versions had a Sentence-BERT fallback here, but it's now disabled

    # Fallback option (currently commented out):
    # If all LLM-guided rounds fail, fall back to semantic similarity search
    # retrieved_doc = find_best_docstring(doc_path, user_query, top_k=1)[0] or []
    # return retrieved_doc

    # Current behavior: Return None to indicate failure
    _print_total_cost(cost_tracker, max_rounds)
    return None
