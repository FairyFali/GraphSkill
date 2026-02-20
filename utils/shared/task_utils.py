"""
Task Utilities: Task loading and processing functions

This module provides utilities for loading task descriptions and
processing task-related data.

Functions:
    load_task_description(task_name, base_path): Load task description from file
"""

import pathlib
from typing import Optional


def load_task_description(task_name: str, base_path: Optional[pathlib.Path] = None) -> str:
    """
    Load task description from text file.

    Args:
        task_name: Name of the task (without .txt extension)
        base_path: Base directory for task descriptions
                  (default: evaluation_dataset/graphtutor_dataset_scripts/task_descriptions)

    Returns:
        Task description text

    Raises:
        FileNotFoundError: If task description file doesn't exist

    Example:
        desc = load_task_description("clustering_and_shortest_path")
    """
    if base_path is None:
        base_path = pathlib.Path("evaluation_dataset/graphtutor_dataset_scripts/task_descriptions")

    task_file = base_path / f"{task_name}.txt"

    if not task_file.exists():
        raise FileNotFoundError(f"Task description not found: {task_file}")

    return task_file.read_text(encoding="utf-8").strip()
