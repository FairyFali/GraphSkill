"""
Config Loader: Load task configuration from JSON files

This module provides utilities for loading task configurations from
JSON files in the configs/tasks/ directory.

Functions:
    load_task_config(config_name): Load task configuration by name
    load_graphtutor_composite_tasks(): Load GraphTutor composite task configuration
    load_graphtutor_standard_tasks(): Load GraphTutor standard task configuration
    load_gtools_tasks(): Load GTools task configuration
"""

import json
import pathlib
from typing import Dict, Any


def load_task_config(config_name: str) -> Dict[str, Any]:
    """
    Load task configuration from JSON file.

    Args:
        config_name: Name of config file without .json extension

    Returns:
        Dictionary containing task configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON

    Example:
        config = load_task_config("graphtutor_composite_tasks")
    """
    config_path = pathlib.Path(__file__).parent.parent.parent / "configs" / "tasks" / f"{config_name}.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return json.loads(config_path.read_text(encoding="utf-8"))


def load_graphtutor_composite_tasks() -> Dict[str, Any]:
    """
    Load GraphTutor composite task configuration.

    Returns:
        Dictionary containing composite task configuration

    Example:
        tasks = load_graphtutor_composite_tasks()
        for graph_type, config in tasks.items():
            print(config['tasks'])
    """
    return load_task_config("graphtutor_composite_tasks")


def load_graphtutor_standard_tasks() -> Dict[str, Any]:
    """
    Load GraphTutor standard task configuration.

    Returns:
        Dictionary containing standard task configuration

    Example:
        tasks = load_graphtutor_standard_tasks()
    """
    return load_task_config("graphtutor_standard_tasks")


def load_gtools_tasks() -> Dict[str, Any]:
    """
    Load GTools task configuration.

    Returns:
        Dictionary containing GTools task configuration

    Example:
        tasks = load_gtools_tasks()
    """
    return load_task_config("gtools_tasks")
