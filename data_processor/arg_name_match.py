import re
from collections import OrderedDict
from file_processing_helpers.find_files import find_files
import os
from pathlib import Path
import jsonlines

# def process_record(record: dict, root: str):
#     argument_dict = {}
#     task_code = root.split("/")[-2]
#     for _, subdict in record.items():
#         argument_dict[task_code] = list(subdict.keys())
#     return argument_dict


def read_txt_files_into_dict(folder_path):
    txt_dict = {}
    # Iterate over all files in the specified folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a .txt file
        if file_name.endswith(".txt"):
            # Build the full file path
            file_path = os.path.join(folder_path, file_name)
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            # Remove the .txt extension to use as dictionary key
            key_name = os.path.splitext(file_name)[0]
            txt_dict[key_name] = content
    return txt_dict

DES_BASE_FOLDER = "../evaluation_dataset/dataset/task_description"
GRAPH_BASE_FOLDER = "../evaluation_dataset/dataset/data"
is_directed_list = ["directed", "undirected"]

description_dict = {}
for is_directed_str in is_directed_list:
    description_base_path = os.path.join(DES_BASE_FOLDER, is_directed_str)
    description_dict.update(read_txt_files_into_dict(description_base_path))

updated_desc_dict = {}
for task_code, description in description_dict.items():
    # --- core logic --------------------------------------------------------
    words = OrderedDict()                       # preserves first-seen order
    for chunk in re.findall(r'Input:(.*?)Output:', description, flags=re.S):
        chunk = re.sub(r'"[^"]*"', '', chunk)                  # ← strip "quoted stuff"
        for w in re.findall(r'[A-Za-z]+', chunk):              # letters only
            words[w] = None                         # OrderedDict ⇒ no duplicates

    result = list(words)      # ['richer', 'quiet']  for the sample paragraph
    updated_desc_dict[task_code]=result

print(updated_desc_dict)

argument_dict = {}
graph_paths = find_files(GRAPH_BASE_FOLDER, ".jsonl")
for graph_path in graph_paths:
    try:
        task_code = graph_path.split("/")[-2]
        with jsonlines.open(graph_path, mode="r") as reader:
            for record in reader:     
                for _, subdict in record.items():
                    subdict.pop("labels", None) or subdict.pop("label", None)
                    subdict.pop("complexity", None)
                    argument_dict[task_code] = list(subdict.keys())
    except Exception as e:
        print(f"Error {e} occured during execution")

print(argument_dict)
print(len(argument_dict))

RENAME_MAP = {
 'lc207': {'graphs': 'prerequisites', 'numCourse': 'numCourses'},
 'lc210': {'graphs': 'prerequisites', 'numCourse': 'numCourses'},
 'lc743': {'graphs': 'times', 'n': 'n', 'k': 'k'},
 'lc797': {'graphs': 'graph'},
 'lc802': {'graphs': 'graph'},
 'lc851': {'graphs': 'richer', 'quiet': 'quiet'},
 'lc997': {'graphs': 'trust', 'n': 'n'},
 'lc1192': {'graphs': 'connections', 'numCourse': 'n'},
 'lc1319': {'graphs': 'connections', 'numCourse': 'n'},
 'lc1462': {'graphs': 'prerequisites', 'n': 'numCourses', 'query': 'queries'},
 'lc1466': {'graphs': 'connections', 'n': 'n'},
 'lc1557': {'graphs': 'edges', 'n': 'n'},
 'lc1615': {'graphs': 'roads', 'numCourse': 'n'},
 'lc1719': {'graphs': 'pairs'},
 'lc1761': {'graphs': 'edges', 'n_list': 'n'},
 'lc1782': {'graphs': 'edges', 'numNodes': 'n', 'queries': 'queries'},
 'lc1971': {'graphs': 'edges', 'n_list': 'n', 'source': 'source', 'destination': 'destination'},
 'lc2050': {'graphs': 'relations', 'n': 'n', 'time': 'time'},
 'lc2097': {'graphs': 'pairs'},
 'lc2192': {'graphs': 'edgeList', 'n': 'n'},
 'lc2242': {'graphs': 'edges', 'scores': 'scores'},
 'lc2246': {'graphs': 'parent', 's': 's'},
 'lc2285': {'graphs': 'roads', 'numNodes': 'n'},
 'lc2316': {'graphs': 'edges', 'numNodes': 'n'},
 'lc2328': {'graphs': 'grid'},
 'lc2360': {'graphs': 'edges'},
 'lc2368': {'graphs': 'edges', 'numNodes': 'n', 'restricted': 'restricted'},
 'lc2374': {'graphs': 'edges'},
 'lc2421': {'graphs': 'edges', 'values': 'vals'},
 'lc2467': {'graphs': 'edges', 'amount': 'amount', 'bob': 'bob'},
 'lc2477': {'graphs': 'roads', 'seats': 'seats'},
 'lc2493': {'graphs': 'edges', 'n': 'n'},
 'lc2508': {'graphs': 'edges', 'n': 'n'},
 'lc2603': {'graphs': 'edges', 'coins': 'coins'},
 'lc2646': {'graphs': 'edges', 'numNodes': 'n', 'price': 'price', 'trips': 'trips'},
 'lc2685': {'graphs': 'edges', 'n': 'n'},
 'lc2858': {'graphs': 'edges', 'n': 'n'},
 'lc2876': {'graphs': 'edges'},
 'lc2924': {'graphs': 'edges', 'n': 'n'},
 'lc3017': {'n': 'n', 'x': 'x', 'y': 'y'}
}

def rename_and_reorder(inner: dict, task_code: str) -> OrderedDict:
    """
    Return a new OrderedDict whose keys are
    (1) renamed via RENAME_MAP (defaulting to original if not found) and
    (2) arranged according to NEW_KEY_ORDER, with any leftovers appended.
    """
    # First, rename
    renamed = {
        RENAME_MAP[task_code].get(k, k): v
        for k, v in inner.items()
    }

    # Then, order deterministically
    ordered = OrderedDict()
    for k in updated_desc_dict[task_code]:
        if k in renamed:
            ordered[k] = renamed.pop(k)      # remove so we don't add twice
    # Append leftover keys (stable iteration order from Python ≥3.7)
    ordered.update(renamed)
    return ordered
for graph_path in graph_paths:
    try:
        with jsonlines.open(graph_path, "r") as reader, jsonlines.open("../evaluation_dataset/dataset/data_renamed/"+graph_path.split("/")[-3]+"/"+graph_path.split("/")[-2]+"/"+graph_path.split("/")[-1], "w") as writer:
            task_code = graph_path.split("/")[-2]
            for obj in reader:
                # obj is the outer dict: {key1:{...}, key2:{...}, ...}
                for outer_key, inner_dict in obj.items():
                    obj[outer_key] = rename_and_reorder(inner_dict, task_code)
                writer.write(obj)
    except Exception as e:
        print(f"{e} occured skipping file {graph_path}")