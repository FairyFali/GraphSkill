"""
Utility functions for ComplexGraph coding agent experiments.

This module provides reusable functions for:
- Creating code generation prompts with retrieved documentation
- Creating error correction prompts for debugging
- Loading test cases from JSON files

These functions are shared across different retrieval methods (TF-IDF, Sentence-BERT, Retrieval Agent).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def create_code_generation_prompt(
    question_text: str,
    is_weighted: bool,
    is_directed: bool,
    args: Optional[Dict[str, Any]],
    retrieved_docs: List[str],
    retrieval_method: str = "RAG",
    test_case: Optional[Dict[str, Any]] = None
) -> str:
    '''
    not using test_case arg. 
    '''
    # Format graph properties
    directed_text = "directed" if is_directed else "undirected"
    weighted_text = "weighted" if is_weighted else "unweighted"

    # Format NetworkX graph class
    nx_graph_class = "nx.DiGraph()" if is_directed else "nx.Graph()"

    # Format arguments if present
    args_desc = "None"
    if args:
        args_list = [f"{k}={v}" for k, v in args.items()]
        args_desc = f"Additional parameters: {', '.join(args_list)}"

    # Format retrieved documentation
    docs_section = ""
    if retrieved_docs:
        docs_section = f"\n\n# Retrieved NetworkX Documentation ({retrieval_method}):\n"
        docs_section += "\n---\n".join(f"Doc {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs))

    # Format test case if provided (currently commented out in original code)
    test_case_section = ""
    # Uncomment below if you want to include test case in prompt
    # if test_case:
    #     test_case_section = f"""
    #
    # # Test Case (for validation):
    # Input:
    #   edge_list = {test_case.get('edge_list', [])}
    #   {' '.join(f"{k} = {v}" for k, v in test_case.get('args', {}).items())}
    #
    # Expected Output: {test_case.get('answer', 'N/A')}
    #
    # Note: Test your function with this input and ensure it produces the expected output.
    # Print the output to verify correctness."""

    prompt = f"""You are tasked with implementing a Python function to solve a graph problem.

Here is relevant NetworkX documentation that may help you solve this problem:

{docs_section}

# Task Description:
{question_text}

Graph Properties:
- Type: {directed_text}, {weighted_text}

Input:
- edge_list: A {weighted_text} {directed_text} graph represented as a list of edges.
  - If weighted: Each edge is [source, target, weight] where weight is a float.
  - If unweighted: Each edge is [source, target].
- {args_desc}


# Requirements:
1. Implement a function that takes an edge list representation of the graph
2. Convert the edge list to a NetworkX {nx_graph_class} object
3. Use NetworkX functions to solve the task
4. Return the result as specified in the task description
5. Return None if any error occurs during execution

# Important Notes:
- CAREFULLY read the NetworkX documentation provided above
- Pay attention to parameter names, types, and return values
- Handle edge cases (empty graphs, disconnected components, etc.)
- Use try-except blocks to catch errors and return None on failure
{test_case_section}

# Your Task:
Generate a complete Python function that solves this problem. Include:
1. Necessary imports (networkx as nx, etc.)
2. Function definition with clear parameter names
3. Edge list to NetworkX graph conversion
4. NetworkX operations to solve the task
5. Return statement with the correct result type
6. Error handling (return None on errors)

Provide your implementation in a Python code block:
"""

    return prompt


def create_error_correction_prompt(
    original_query: str,
    error_code: str,
    error_output: str,
    test_case: Dict[str, Any]
) -> str:
    """
    Create prompt for error correction (debugging).

    Args:
        original_query: Original task description
        error_code: The code that failed
        error_output: Error message or incorrect output
        test_case: Test case with 'edge_list', 'args', and 'answer' fields

    Returns:
        Formatted prompt string for error correction

    Example:
        >>> prompt = create_error_correction_prompt(
        ...     original_query="Find shortest path...",
        ...     error_code="def shortest_path(edge_list): ...",
        ...     error_output="KeyError: 'source'",
        ...     test_case={'edge_list': [[0,1]], 'args': {'source': 0}, 'answer': 1}
        ... )
    """
    # Format test case input for display
    if test_case: 
        test_input_str = f"""edge_list = {test_case.get('edge_list', [])}"""
        if test_case.get('args'):
            for k, v in test_case['args'].items():
                test_input_str += f"\n{k} = {v}"

        prompt = f"""The following code failed to produce the correct output for a graph problem.

            Original Task:
            {original_query}

            Failed Code:
            ```python
            {error_code}
            ```

            Error/Incorrect Output:
            {error_output}

            Test Case:
            Input:
            {test_input_str}

            Expected Output: {test_case.get('answer', 'N/A')}

            Analyze the error and provide a corrected version of the code that:
            1. Fixes any syntax or runtime errors
            2. Produces the correct output matching the expected result
            3. Handles edge cases properly
            4. Returns None if any error occurs
            5. Test with the provided input and print the output

            Provide the corrected implementation in a Python code block:
            """
    else: 

        prompt = f"""The following code failed to produce the correct output for a graph problem.

            Original Task:
            {original_query}

            Failed Code:
            ```python
            {error_code}
            ```

            Error/Incorrect Output:
            {error_output}

            Analyze the error and provide a corrected version of the code that:
            1. Fixes any syntax or runtime errors
            2. Produces the correct output matching the expected result
            3. Handles edge cases properly
            4. Returns None if any error occurs
            5. Test with the provided input and print the output

            Provide the corrected implementation in a Python code block:
            """


    return prompt


def load_test_case_from_file(
    test_case_path: str,
    task_name: str,
    is_directed: bool,
    is_weighted: bool
) -> Optional[Dict[str, Any]]:
    """
    Load test case from JSON file based on task properties.

    Args:
        test_case_path: Base path to test cases directory
        task_name: Name of the task (e.g., "clustering", "shortest_path_length")
        is_directed: Whether the graph is directed
        is_weighted: Whether the graph is weighted

    Returns:
        Dictionary with 'edge_list', 'args', and 'answer' fields, or None if not found

    Example:
        >>> test_case = load_test_case_from_file(
        ...     test_case_path="prompts/graph_tasks_testing_cases/",
        ...     task_name="clustering",
        ...     is_directed=True,
        ...     is_weighted=True
        ... )
        >>> print(test_case)
        {'edge_list': [[0,1,0.5]], 'args': {}, 'answer': 0.123}
    """
    try:
        # Build path based on graph properties
        direction_str = "directed" if is_directed else "undirected"
        weight_str = "weighted" if is_weighted else "unweighted"

        test_case_file = Path(test_case_path) / direction_str / weight_str / "test_cases.json"

        if not test_case_file.exists():
            print(f"⚠ Warning: Test case file not found: {test_case_file}")
            return None

        with open(test_case_file, 'r') as f:
            test_case_data = json.load(f)

        if task_name not in test_case_data:
            print(f"⚠ Warning: Test case for '{task_name}' not found in {test_case_file}")
            return None

        return test_case_data[task_name]

    except Exception as e:
        print(f"⚠ Warning: Error loading test case: {e}")
        return None


def format_test_case_info(test_case: Dict[str, Any]) -> str:
    """
    Format test case information for display.

    Args:
        test_case: Test case dictionary with 'edge_list', 'args', and 'answer' fields

    Returns:
        Formatted string describing the test case

    Example:
        >>> test_case = {'edge_list': [[0,1]], 'args': {'source': 0}, 'answer': 1}
        >>> print(format_test_case_info(test_case))
        1 edges, args: {'source': 0}, expected answer: 1
    """
    edge_count = len(test_case.get('edge_list', []))
    args = test_case.get('args', {})
    answer = test_case.get('answer')

    return f"{edge_count} edges, args: {args}, expected answer: {answer}"
