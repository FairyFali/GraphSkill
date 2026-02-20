#!/usr/bin/env python3
"""
Read file.jsonl (JSON-Lines format) where every line looks like
{"<top_key>": {"<sub_key1>": <value1>, "<sub_key2>": <value2>, …}}
and create

file/<top_key>/<sub_key1>.txt   containing <value1>
file/<top_key>/<sub_key2>.txt   containing <value2>
…

Usage:
    python build_files.py          # assumes file.jsonl in the same folder
"""
import jsonlines
import json
import pathlib
import sys
import os
from pathlib import Path
from file_processing_helpers.find_files import find_files

BASE_PATH   = pathlib.Path("../evaluation_dataset/dataset/data")   # jsonl source  


def write_value(root: Path, outer_key: str, idx: int, inner_key: str, value):
    path = root / outer_key / str(idx) / f"{inner_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)

def process_record(record: dict, root: Path):
    for outer_key, subdict in record.items():
        max_len = max(len(lst) for lst in subdict.values())
        for i in range(max_len):
            for inner_key, lst in subdict.items():
                if i < len(lst):
                    write_value(root, outer_key, i, inner_key, lst[i])


def main(jsonl_file):
    if not pathlib.Path(jsonl_file).exists():
        sys.exit(f"Input file {jsonl_file} not found.")

    jsonl_file_name = jsonl_file.split("/")[-1]
    file_base_path = jsonl_file[:-(len(jsonl_file_name)+1)]
    output_dir_name = jsonl_file_name.split(".")[0]
    output_path = Path(os.path.join(file_base_path, output_dir_name))
    try:
        with jsonlines.open(jsonl_file, mode="r") as reader:
            for record in reader:     
                process_record(record, output_path)
    except Exception as e:
        print(f"Error {e} occured during execution")


if __name__ == "__main__":
    graph_jsonl_files = find_files(BASE_PATH, ".jsonl")
    for jsonl_file in graph_jsonl_files:
        main(jsonl_file)
