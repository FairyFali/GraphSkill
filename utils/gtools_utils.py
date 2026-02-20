"""
GTools Dataset Utilities

This module provides utility functions for working with the GTools dataset
in the converted ComplexGraph format.

Functions:
    - load_gtools_data(): Load GTools questions and graphs from JSON files
    - Additional GTools-specific utilities

Example:
    from utils.gtools_utils import load_gtools_data

    questions_data, graphs_data = load_gtools_data('small')
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Constants
GTOOLS_BASE_PATH = Path("/data/faliwang/GTools")


def load_gtools_data(dataset_version: str) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Load GTools dataset in ComplexGraph format.

    The GTools dataset has been converted to match ComplexGraph structure:
    - questions.json: Contains task metadata and test instances
    - graphs.json: Contains graph structures indexed by unique IDs

    Args:
        dataset_version: Dataset version ('small' or 'large')

    Returns:
        Tuple of (questions_data, graphs_data) where:
            - questions_data: List of question groups with task info and instances
            - graphs_data: Dictionary mapping graph IDs to graph structures

    Raises:
        FileNotFoundError: If questions.json or graphs.json not found
        ValueError: If dataset_version is invalid

    Example:
        >>> questions, graphs = load_gtools_data('small')
        ✓ Loaded GTools-SMALL dataset
          - Questions: 42 groups
          - Graphs: 5000 instances
        >>> print(len(questions))
        42
        >>> print(len(graphs))
        5000
    """
    # Validate dataset version
    if dataset_version not in ['small', 'large']:
        raise ValueError(f"Invalid dataset version: {dataset_version}. Must be 'small' or 'large'.")

    dataset_path = GTOOLS_BASE_PATH / dataset_version

    # Load questions.json
    questions_file = dataset_path / "questions.json"
    if not questions_file.exists():
        raise FileNotFoundError(
            f"Questions file not found: {questions_file}\n"
            f"Please ensure the GTools dataset has been converted to ComplexGraph format.\n"
            f"Run: python convert_gtool_data.py --dataset {dataset_version}"
        )

    with open(questions_file, "r") as f:
        questions_data = json.load(f)

    # Load graphs.json
    graphs_file = dataset_path / "graphs.json"
    if not graphs_file.exists():
        raise FileNotFoundError(
            f"Graphs file not found: {graphs_file}\n"
            f"Please ensure the GTools dataset has been converted to ComplexGraph format.\n"
            f"Run: python convert_gtool_data.py --dataset {dataset_version}"
        )

    with open(graphs_file, "r") as f:
        graphs_data = json.load(f)

    print(f"✓ Loaded GTools-{dataset_version.upper()} dataset")
    print(f"  - Questions: {len(questions_data)} groups")
    print(f"  - Graphs: {len(graphs_data)} instances\n")

    return questions_data, graphs_data


def get_gtools_dataset_path(dataset_version: str) -> Path:
    """
    Get the path to a GTools dataset.

    Args:
        dataset_version: Dataset version ('small' or 'large')

    Returns:
        Path to the dataset directory

    Example:
        >>> path = get_gtools_dataset_path('small')
        >>> print(path)
        /data/faliwang/GTools/small
    """
    if dataset_version not in ['small', 'large']:
        raise ValueError(f"Invalid dataset version: {dataset_version}. Must be 'small' or 'large'.")

    return GTOOLS_BASE_PATH / dataset_version


def validate_gtools_dataset(dataset_version: str) -> bool:
    """
    Check if a GTools dataset is properly formatted and exists.

    Args:
        dataset_version: Dataset version ('small' or 'large')

    Returns:
        True if dataset is valid, False otherwise

    Example:
        >>> if validate_gtools_dataset('small'):
        ...     print("Dataset is ready!")
        Dataset is ready!
    """
    try:
        dataset_path = get_gtools_dataset_path(dataset_version)

        # Check if directory exists
        if not dataset_path.exists():
            print(f"✗ Dataset directory not found: {dataset_path}")
            return False

        # Check for required files
        questions_file = dataset_path / "questions.json"
        graphs_file = dataset_path / "graphs.json"

        if not questions_file.exists():
            print(f"✗ Missing questions.json in {dataset_path}")
            return False

        if not graphs_file.exists():
            print(f"✗ Missing graphs.json in {dataset_path}")
            return False

        # Try to load and validate basic structure
        with open(questions_file, "r") as f:
            questions = json.load(f)
            if not isinstance(questions, list):
                print(f"✗ Invalid questions.json format (not a list)")
                return False

        with open(graphs_file, "r") as f:
            graphs = json.load(f)
            if not isinstance(graphs, dict):
                print(f"✗ Invalid graphs.json format (not a dict)")
                return False

        print(f"✓ GTools-{dataset_version.upper()} dataset is valid")
        return True

    except Exception as e:
        print(f"✗ Error validating dataset: {e}")
        return False
