import ast
import json
import os
import pathlib
import re, ast, textwrap, types

from networkx import shortest_path_length
from data_pipeline.dataset_generation.graph_dataset_gen import save_dict_to_json
from utils.get_llm_response_generator import create_code_generator
from utils.generation_functions.generate_prompt import generate_code_with_openai

all_tasks = {
    "weighted_undirected_tasks": {
        "tasks": ["clustering",  "shortest_path_length", "maximum_flow"],    #finished: 
        "args": {"common_neighbors": ["source", "target"], "maximum_flow":["_snode", "_tnode"], "traveling_salesman_problem": ["nodes"], "connected_components":None, "max_clique": None, "shortest_path_length": ["source", "target"], "clustering": ["node"], "diameter": None},
        "inputs": {"maximum_flow": "int or float or str", "common_neighbors":"int or float or str","traveling_salesman_problem":"List[Int] or List[Float] or List[Str]", "connected_components":None, "max_clique": None, "shortest_path_length": "int or float or str", "clustering": "int or float or str", "diameter": None},
        "returns": {"maximum_flow": "int or float", "common_neighbors":"List[Int] or List[Float] or List[Str]", "traveling_salesman_problem": "List[Int] or List[Float] or List[Str]","connected_components":"List[Int] or List[Float] or List[Str]", "max_clique": "List[Int] or List[Float] or List[Str]","shortest_path_length": "int", "clustering": "float", "diameter": "int"},
        "directed": False,
        "weighted": True
    },
    "unweighted_directed_tasks": {
        "tasks": ["clustering"], #finished: , "is_bipartite", "has_eulerian_path", "topological_sort", "strongly_connected_components", "is_regular", "diameter"
        "args": { "diameter": None, "is_regular": None, "clustering": ["node"], "traveling_salesman_problem": ["nodes"], "min_weighted_vertex_cover": None,"is_bipartite": None, "hamiltonian_path": None, "strongly_connected_components": None, "has_eulerian_path": None, "topological_sort": None},
        "inputs": {"diameter": None, "is_regular": None, "clustering": "int or float or str", "traveling_salesman_problem":"List[Int] or List[Float] or List[Str]", "min_weighted_vertex_cover": None, "is_bipartite": None, "hamiltonian_path": None, "strongly_connected_components": None, "has_eulerian_path": None, "topological_sort": None},
        "returns": {"diameter": "int", "is_regular": "bool", "clustering": "float", "traveling_salesman_problem": "List[Int] or List[Float] or List[Str]", "min_weighted_vertex_cover":"List[Int] or List[Float] or List[Str]", "is_bipartite": "bool", "hamiltonian_path": "List[Int]", "strongly_connected_components": "List[Int]", "has_eulerian_path": "bool", "topological_sort": "List[Int]"},
        "directed": True,
        "weighted": False
    },
    "weighted_directed_tasks": {
        "tasks": ["maximum_flow", "shortest_path_length", "clustering"], #finished:  
        "args": {"clustering": ["node"], "traveling_salesman_problem": ["nodes"], "maximum_flow":["_snode", "_tnode"], "shortest_path_length": ["source", "target"]},
        "inputs": {"clustering": "int or float or str", "traveling_salesman_problem":"List[Int] or List[Float] or List[Str]", "maximum_flow": "int or float or str", "shortest_path_length": "int or float or str"},
        "returns": {"clustering": "float", "traveling_salesman_problem": "List[Int] or List[Float] or List[Str]","maximum_flow": "int or float", "shortest_path_length": "int"},
        "directed": True,
        "weighted": True
    },
    "unweighted_undirected_tasks": {
        "tasks": ["clustering", "common_neighbors"], #finished: "is_bipartite", "diameter", "is_regular", "is_distance_regular", "is_connected", "has_eulerian_path", "connected_components", , "max_clique", "maximum_independent_set"
        "args": {"is_bipartite": None, "maximum_independent_set":None,"max_clique": None, "connected_components":None,"min_weighted_vertex_cover": None,"common_neighbors": ["source", "target"],"clustering": ["node"], "topological_sort": None, "diameter": None, "is_regular": None, "is_distance_regular": None, "is_connected": None, "has_eulerian_path": None},
        "inputs":  {"is_bipartite": None, "maximum_independent_set":None,"max_clique": None,"connected_components":None,"min_weighted_vertex_cover": None,"common_neighbors":"int or float or str","clustering": "int or float or str", "topological_sort": None, "diameter": None, "is_regular": None, "is_distance_regular": None, "is_connected": None, "has_eulerian_path": None},
        "returns": {"is_bipartite": "bool", "maximum_independent_set":"List[Int] or List[Float] or List[Str]","max_clique": "List[Int] or List[Float] or List[Str]", "connected_components":"List[Int] or List[Float] or List[Str]","min_weighted_vertex_cover":"List[Int] or List[Float] or List[Str]","common_neighbors":"List[Int] or List[Float] or List[Str]", "clustering": "float", "topological_sort": "List[Tuple[Int]]", "diameter": "int", "is_regular": "bool", "is_distance_regular": "bool", "is_connected": "bool", "has_eulerian_path": "bool"},
        "directed": False,
        "weighted": False
    }
}

# all_tasks = {
#     "unweighted_undirected_tasks": {
#         "tasks": ["clustering_and_shortest_path", "highest_clustered_node_in_shortest_path", "shortest_path_and_eular_tour"], # "largest_component_and_diameter",
#         "args": {"clustering_and_shortest_path": ["k"],  "highest_clustered_node_in_shortest_path": ["s", "t"], "largest_component_and_diameter": None, "regular_and_distance_regular": None, "shortest_path_and_eular_tour": ["s", "t"]},
#         "inputs": {"clustering_and_shortest_path": "int",  "highest_clustered_node_in_shortest_path": "int or str or float", "largest_component_and_diameter": None, "regular_and_distance_regular": None, "shortest_path_and_eular_tour": "int or str or float"},
#         "returns": {"clustering_and_shortest_path": "List[Int] or List[Float] or List[Str]", "highest_clustered_node_in_shortest_path": "List[Int] or List[Float] or List[Str]", "largest_component_and_diameter": "int", "regular_and_distance_regular": "bool", "shortest_path_and_eular_tour": "int"},
#         "directed": False,
#         "weighted": False
#     },
#     "weighted_undirected_tasks": {
#         "tasks": ["pair_tightness_score", "flow_aware_local_clustering_among_cc"],
#         "args": {"pair_tightness_score": ["s", "t"], "flow_aware_local_clustering_among_cc": ["t"]},
#         "inputs": {"pair_tightness_score": "int", "flow_aware_local_clustering_among_cc": "int"},
#         "returns": {"pair_tightness_score": "float", "flow_aware_local_clustering_among_cc": "int"},
#         "directed": False,
#         "weighted": True
#     },
#     "weighted_directed_tasks": {
#         "tasks": ["endpoint_aware_flow_score"],
#         "args": {"endpoint_aware_flow_score": ["_snode", "_tnode"]},
#         "inputs": {"endpoint_aware_flow_score": "int"},
#         "returns": {"endpoint_aware_flow_score": "float"},
#         "directed": True,
#         "weighted": True
#     },
#     "unweighted_directed_tasks": {
#         "tasks": [], # "scc_and_diameter_nested", "scc_and_eulerian_feasibility"
#         "args": {"scc_and_diameter_nested": None, "scc_and_eulerian_feasibility": None},
#         "inputs": {"scc_and_diameter_nested": None, "scc_and_eulerian_feasibility": None},
#         "returns": {"scc_and_diameter_nested": "int", "scc_and_eulerian_feasibility": "List[Int]"},
#         "directed": True,
#         "weighted": False
#     },
# }

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


arg_code_dir = pathlib.Path("evaluation_dataset/graphtutor_dataset_scripts/add_args_scripts")

base_data_dir = pathlib.Path("/data/chenglin/graphtutor_graph_dataset") ###REMEMBER TO Change
output_data_dir = pathlib.Path("/data/chenglin/graphtutor_args_labelled")

def main(graph_rep_types):
    for graph_type, config in all_tasks.items():
        print(f"\n{'='*10}")
        print(f"Processing: {graph_type}")
        print(f"Properties: directed={config['directed']}, weighted={config['weighted']}")
        print(f"Tasks: {config['tasks']}")
        print('='*10)
        if graph_type == "weighted_undirected_tasks":
            is_directed = "undirected"
            is_weighted = "weighted"
        elif graph_type == "unweighted_directed_tasks":
            is_directed = "directed"
            is_weighted = "unweighted"
        elif graph_type == "weighted_directed_tasks":
            is_directed = "directed"
            is_weighted = "weighted"
        elif graph_type == "unweighted_undirected_tasks":
            is_directed = "undirected"
            is_weighted = "unweighted"
        else:
            raise ValueError(f"Invalid graph type: {graph_type}")
        for task_name in config['tasks']:
            print(f"\n--- Task: {task_name} ---")
            for graph_rep_type in graph_rep_types:
                base_graph_path = base_data_dir / is_directed / is_weighted / graph_rep_type # change this 
                graph_file_paths = find_graph_json_files(base_graph_path)
                # Generate graphs for this category
                arg_code_path = arg_code_dir / graph_rep_type / is_directed/ is_weighted / "extra_args_scripts.json"
                
                with open(arg_code_path, "r") as f:
                    task_arg_code_dict = json.load(f)
                sample_code = task_arg_code_dict[task_name]
                
                for graph_path in graph_file_paths:
                    graph_json_file_name = graph_path.split("/")[-1]
                    updated_graph_info_dict = {}
                    with open(graph_path, "r") as f:
                        graph_dict = json.load(f)
                    for key in list(graph_dict.keys()):
                        graph_list = graph_dict[key]
                        arg_list = config["args"][task_name]
                        args_dict = {}
                        for arg_name in arg_list:
                            args_dict[arg_name] = []
                        updated_data_dict = {}
                        updated_graph_list = []
                        
                        for graph in graph_list:
                            test_arg_dict = {}
                            updated_graph_list.append(graph)
                            for idx in range(1):
                                try:
                                    label = run_code_for_given_inputs(sample_code, graph)
                                except Exception as e:
                                    import pdb;pdb.set_trace()
                                for i in range(len(arg_list)):
                                    if isinstance(label, tuple):
                                        test_arg_dict[arg_list[i]] = label[i]
                                    else:
                                        test_arg_dict[arg_list[i]] = label
                                    
                            for i in range(len(arg_list)):
                                # arg_list[i], i th arg name in arg list
                                args_dict[arg_list[i]].append(test_arg_dict[arg_list[i]])
                        # import pdb;pdb.set_trace()    
                        updated_data_dict["graphs"] = updated_graph_list
                        updated_graph_info_dict[key] = updated_data_dict | args_dict
                                   
                # Save results
                    output_path = output_data_dir / graph_rep_type / is_directed / is_weighted / task_name / graph_json_file_name
                    save_dict_to_json(updated_graph_info_dict, output_path)
                    print(f"    Saved to: {output_path}")
                
if __name__ == "__main__":
    graph_rep_types = ["edge_list"] # , "adj_list"
    main(graph_rep_types)
