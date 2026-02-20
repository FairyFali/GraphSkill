from pathlib import Path
import json
import re, ast, textwrap, types
import os

BASE_DIR = "../evaluation_dataset/GWild"
code_path = "../evaluation_dataset/GWild_docs/gt_code_for_labels/tasks_require_extra_args.json"

def save_dict_to_json(data: dict, file_path: str | Path, *, indent: int = 4) -> None:
    """
    Serialize a Python dictionary to a JSON file, creating any
    intermediate directories that don’t yet exist.

    Parameters
    ----------
    data : dict
        The dictionary you want to save.
    file_path : str | Path
        Destination path (e.g. "output/settings/config.json").
    indent : int, optional
        Pretty-print spacing in the resulting file (default = 4).
    """
    file_path = Path(file_path)                 # allow strings or Path objects

    # Ensure parent directories exist (no error if they’re already there)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the JSON file with UTF-8 encoding
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def find_graph_json_files(base_folder):
    """
    Recursively walks through the given folder and returns a list of files
    that end with '_{task_name}_quetions.json'.
    """
    question_files = []
    for root, dirs, files in os.walk(base_folder):
        for file_name in files:
            if file_name.endswith(f'.json'):
                # If you only want to print them, you could do:
                # print(os.path.join(root, file_name))

                # Here we'll collect them in a list:
                question_files.append(os.path.join(root, file_name))
                # info_files.append(os.path.join(root, root.split('/')[-1]+".json")) # extract the dataset name from '{dataset_name}_shortest_path_questions.json' file
    return question_files

def run_code_for_given_inputs(md_string, graph_input):
    # ------------------------------------------------------------
    # 1  Pull out the literal code
    code_escaped = re.search(r"```python(.*?)```$", md_string, re.S).group(1)
    code = textwrap.dedent(code_escaped.replace(r"\n", "\n"))

    # ------------------------------------------------------------
    # 2  Parse with AST to discover *top-level* function names
    parsed = ast.parse(code)
    func_names = [
        node.name for node in parsed.body
        if isinstance(node, ast.FunctionDef)
    ]

    if not func_names:
        raise ValueError("No top-level functions found")

    # For demo we’ll just take the first one
    target_name = func_names[0]
    print("Discovered function:", target_name)

    # ------------------------------------------------------------
    # 3  Execute the code in a fresh module namespace
    module = types.ModuleType("dynamic_code")
    exec(code, module.__dict__)          # populates module attributes

    # ------------------------------------------------------------
    # 4  Retrieve & call the discovered function
    target_fn = getattr(module, target_name)
    result = target_fn(graph_input)    # example call
    return result

def main():
    with open(code_path, "r") as f:
        task_name_code_dict = json.load(f)
    for task_name, code in task_name_code_dict.items():
        base_graph_folder_path = os.path.join(BASE_DIR, task_name, "undirected")
        graph_file_dirs = find_graph_json_files(base_graph_folder_path)
        for graph_file_dir in graph_file_dirs:
            with open(graph_file_dir, "r") as f:
                graph_info_dict = json.load(f)
            info_dict_keys = list(graph_info_dict.keys())
            for info_key in info_dict_keys:
                graph_type_dict_keys = graph_info_dict[info_key]
                graph_info_w_args = {}
                for type_key in graph_type_dict_keys:
                    graphs = graph_info_dict[info_key][type_key]
                    graph_para_dict = {}
                    graph_para_dict["graphs"] = graphs
                    if task_name == "edge_boundary":
                        arg1_list = []
                        arg2_list = []
                        for graph in graphs:
                            try:
                                extra_arg1, extra_arg2 = run_code_for_given_inputs(code, graph)
                            except Exception as e:
                                print(f"Error occured {e}")
                                print(f"Adding None to arg list in task: {task_name} graph type: {type_key} graph idx: {graphs.index(graph)} at {graph_file_dir}")
                                extra_arg1 = None
                                extra_arg2 = None
                            arg1_list.append(extra_arg1)
                            arg2_list.append(extra_arg2)
                        graph_para_dict["arg1"] = arg1_list
                        graph_para_dict["arg2"] = arg2_list
                    else:
                        arg_list = []
                        for graph in graphs:
                            try:
                                extra_arg = run_code_for_given_inputs(code, graph)
                            except Exception as e:
                                print(f"Error occured {e}")
                                print(f"Adding None to arg list in task: {task_name} graph type: {type_key} graph idx: {graphs.index(graph)} at {graph_file_dir}")
                                extra_arg = None
                            arg_list.append(extra_arg)
                        graph_para_dict["arg"] = arg_list
                    graph_info_w_args[type_key] = graph_para_dict
                save_dict_to_json(graph_info_w_args, os.path.join("../evaluation_dataset", "nx_func_dataset","undirected", task_name, graph_file_dir.split("/")[-1]))

                            
            
        
if __name__ == '__main__':
    main()