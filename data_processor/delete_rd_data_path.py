#!/usr/bin/env python3
"""
Prune “grand-child” folders one level below each direct sub-directory
of a root directory, while never deleting any .jsonl files.

Usage:
    python prune_subdirs.py [path/to/root_dir]

If no path is supplied, the script works on the current directory.
"""

from pathlib import Path
import shutil
import sys

def contains_jsonl(path: Path) -> bool:
    """Return True if *any* .jsonl file exists under *path* (recursively)."""
    return any(p.suffix == ".jsonl" for p in path.rglob("*.jsonl"))

def prune(root: Path) -> None:
    if not root.is_dir():
        raise NotADirectoryError(root)

    for lvl1 in root.iterdir():
        if not lvl1.is_dir():
            continue  # ignore files at root level

        for candidate in lvl1.iterdir():          # level-2 entries
            if not candidate.is_dir():
                continue

            if contains_jsonl(candidate):
                print(f"SKIP: {candidate}  (contains .jsonl file(s))")
            else:
                print(f"DELETE: {candidate}")
                shutil.rmtree(candidate)

if __name__ == "__main__":
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    prune(root_dir.resolve())
