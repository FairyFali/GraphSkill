"""
JSON Utilities: Safe JSON loading and saving operations

This module provides utilities for safely loading and saving JSON files,
with error handling for corrupt or empty files.

Functions:
    load_json_safe(path): Load JSON with fallback to empty dict on error
    save_dict_to_json(data, path): Save dictionary to JSON file
"""

import json
import pathlib
from typing import Any, Dict, Union


class _ExtendedEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable types like set."""
    def default(self, obj):
        if isinstance(obj, set):
            return sorted(list(obj))
        if isinstance(obj, bool):
            return str(obj)
        return super().default(obj)


def load_json_safe(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """
    Load JSON file with error handling for corrupt or empty files.

    Args:
        path: Path to JSON file (string or Path object)

    Returns:
        Dictionary containing JSON data, or empty dict on error

    Example:
        data = load_json_safe("config.json")
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        # File is empty, corrupt, or doesn't exist → return empty dict
        return {}


def save_dict_to_json(data: Dict[str, Any], path: Union[str, pathlib.Path], indent: int = 2) -> None:
    """
    Save dictionary to JSON file with pretty formatting.

    Args:
        data: Dictionary to save
        path: Output file path (string or Path object)
        indent: JSON indentation level (default: 2)

    Example:
        save_dict_to_json({"key": "value"}, "output.json")
    """
    if type(data) is not dict: 
        # raise ValueError(f"Expected data to be a dict, got {type(data)}")
        print(f"Warning: Expected data to be a dict, got {type(data)}. Skipping save.")
        return
    if isinstance(path, str):
        path = pathlib.Path(path)
    try: 
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=indent, ensure_ascii=False, cls=_ExtendedEncoder), encoding="utf-8")
    except Exception as e:
        print(f"✗ Error saving JSON to {path}: {e}. Data: {data}")