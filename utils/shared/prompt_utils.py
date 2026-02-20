"""
Prompt Utilities: Prompt building and formatting functions

This module provides utilities for building prompts for LLM code generation
and other LLM-based tasks.

Functions:
    build_code_prompt(task_description, graph_properties): Build code generation prompt
"""

from typing import Dict, Optional


def build_code_prompt(
    task_description: str,
    directed: bool,
    weighted: bool,
    additional_context: Optional[str] = None
) -> str:
    """
    Build a code generation prompt for graph tasks.

    Args:
        task_description: Description of the task to implement
        directed: Whether the graph is directed
        weighted: Whether the graph is weighted
        additional_context: Optional additional context or instructions

    Returns:
        Formatted prompt string

    Example:
        prompt = build_code_prompt(
            task_description="Find shortest path",
            directed=False,
            weighted=True
        )
    """
    graph_type_desc = []
    if directed:
        graph_type_desc.append("directed")
    else:
        graph_type_desc.append("undirected")

    if weighted:
        graph_type_desc.append("weighted")
    else:
        graph_type_desc.append("unweighted")

    graph_desc = " and ".join(graph_type_desc)

    prompt = f"""Task: {task_description}

Graph Type: {graph_desc} graph

Please implement a solution for this task."""

    if additional_context:
        prompt += f"\n\n{additional_context}"

    return prompt
