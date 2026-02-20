"""
Shared utility functions for ComplexGraph code generation experiments.

This module contains common functions used across different code generation
approaches (TF-IDF retrieval, Sentence-BERT retrieval, GraphTeam, etc.).
"""

from typing import List, Dict, Any, Optional


def create_code_generation_prompt_with_docs(
    question_text: str,
    is_weighted: bool,
    is_directed: bool,
    args: Optional[Dict[str, Any]],
    retrieved_docs: Optional[List[str]]
) -> str:
    directed_text = "directed" if is_directed else "undirected"
    weighted_text = "weighted" if is_weighted else "unweighted"
    
    # Format NetworkX graph class
    nx_graph_class = "nx.DiGraph()" if is_directed else "nx.Graph()"

    args_desc = ""
    if args:
        args_list = [f"{k} (variable name: '{k}')" for k in args.keys()]
        args_desc = f"The following variables are provided: {', '.join(args_list)}."


    # RAG-enhanced prompt with documentation context
    docs_text = "\n\n" + "="*70 + "\n\n".join(retrieved_docs)
#     prompt = f"""Given the task description: {question_text}

# Generate a Python function that solves this task for a {weighted_text} {directed_text} graph.

# Here is relevant NetworkX documentation that may help you solve this problem:

# {docs_text}

# Input:
# - edge_list: A {weighted_text} {directed_text} graph represented as a list of edges.
# - If weighted: Each edge is [source, target, weight] where weight is a float.
# - If unweighted: Each edge is [source, target].
# - {args_desc}

# Output:
# - Store your final answer in a variable named 'result'.
# - The result should match the expected return type for this task.

# Constraints:
# - ONLY use NetworkX functions mentioned in the provided documentation above.
# - Import NetworkX as: import networkx as nx
# - First convert edge_list to a NetworkX graph object.
# - For directed graphs, use nx.DiGraph().
# - For undirected graphs, use nx.Graph().
# - When adding weighted edges, use: G.add_weighted_edges_from(edge_list)
# - When adding unweighted edges, use: G.add_edges_from(edge_list)
# - Write clean, efficient Python code.
# - Do NOT add example usage or test code.
# - Do NOT print anything.
# - ONLY provide the code, no explanations.

# Your code:
# """
    
    prompt = f"""You are tasked with implementing a Python function to solve a graph problem.

Here is relevant NetworkX documentation that may help you solve this problem:

{docs_text}

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


def format_args_description(args: Optional[Dict[str, Any]]) -> str:
    """
    Format task arguments into a human-readable description.

    Args:
        args: Dictionary of task-specific arguments (e.g., {"node": 5})

    Returns:
        Formatted string describing the arguments

    Example:
        >>> format_args_description({"node": 5})
        'node: 5'
        >>> format_args_description({"source": 1, "target": 3})
        'source: 1, target: 3'
        >>> format_args_description(None)
        'None'
    """
    if not args:
        return "None"

    args_parts = []
    for key, value in args.items():
        args_parts.append(f"{key}: {value}")

    return ", ".join(args_parts)


def extract_python_code(response: str) -> str:
    """
    Extract Python code from LLM response.

    Handles various response formats:
    1. Code blocks with ```python markers
    2. Code blocks with ``` markers
    3. Raw code without markers

    Args:
        response: Raw LLM response text

    Returns:
        Extracted Python code string

    Example:
        >>> extract_python_code("```python\\nimport nx\\nG = nx.Graph()\\n```")
        'import nx\\nG = nx.Graph()'
    """
    response = response.strip()

    # Try to extract from ```python code blocks
    if "```python" in response:
        code_parts = response.split("```python")
        if len(code_parts) > 1:
            code = code_parts[1].split("```")[0].strip()
            return code

    # Try to extract from ``` code blocks
    if "```" in response:
        code_parts = response.split("```")
        if len(code_parts) >= 2:
            # Take the first code block (usually index 1)
            code = code_parts[1].strip()
            # Remove language identifier if present (e.g., "python\n")
            if code.startswith("python\n"):
                code = code[7:]  # len("python\n") = 7
            return code

    # If no code blocks, return the entire response (assume it's raw code)
    return response


def validate_code_syntax(code: str) -> tuple[bool, Optional[str]]:
    """
    Validate Python code syntax without executing it.

    Uses AST compilation to check for syntax errors.

    Args:
        code: Python code string to validate

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])

    Example:
        >>> validate_code_syntax("import networkx as nx\\nG = nx.Graph()")
        (True, None)
        >>> validate_code_syntax("import networkx as nx\\nG = nx.Graph(")
        (False, "SyntaxError: ...")
    """
    import ast

    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def create_error_correction_prompt(
    original_prompt: str,
    generated_code: str,
    error_message: str,
    execution_result: Optional[Any] = None,
    ground_truth: Optional[Any] = None
) -> str:
    """
    Create a prompt for error correction / self-debugging.

    Used in iterative code generation approaches like GraphTeam where the LLM
    can fix errors based on feedback.

    Args:
        original_prompt: The original code generation prompt
        generated_code: The code that was generated
        error_message: Error message or description of the problem
        execution_result: Actual execution result (if code ran)
        ground_truth: Expected result (if available)

    Returns:
        Prompt string for error correction

    Note:
        Used by GraphTeam and other iterative debugging approaches
    """
    correction_prompt = f"""The code you generated has an issue. Please fix it.

Original Task:
{original_prompt}

Your Previous Code:
```python
{generated_code}
```

Problem:
{error_message}
"""

    if execution_result is not None:
        correction_prompt += f"\nYour code returned: {execution_result}"

    if ground_truth is not None:
        correction_prompt += f"\nExpected result: {ground_truth}"

    correction_prompt += """

Please provide corrected code that:
1. Fixes the error
2. Produces the correct result
3. Follows all the original constraints

Corrected code:
"""

    return correction_prompt
