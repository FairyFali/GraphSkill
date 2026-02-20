import json
import pathlib
from data_pipeline.dataset_generation.graph_dataset_gen import save_dict_to_json
from utils.get_llm_response_generator import create_code_generator
from utils.generation_functions.generate_prompt import generate_code_with_openai

    
def load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # maybe file is empty/corrupt → return empty list/dict as you prefer
        return {}

all_tasks = {
    "weighted_undirected_tasks": {
        "tasks": [], #     "max_clique","connected_components", "traveling_salesman_problem", "shortest_path_length", "diameter", "clustering", "common_neighbors" 
        "args": {"common_neighbors": ["graphs", "source", "target"], "traveling_salesman_problem": ["graphs","nodes"], "connected_components":["graphs"], "max_clique": ["graphs"], "shortest_path_length": ["graphs","source", "target"], "clustering": ["graphs", "nodes"], "diameter": ["graphs"]},
        "inputs": {"common_neighbors":"int or float or str","traveling_salesman_problem":"int or float or str", "connected_components":None, "max_clique": None, "shortest_path_length": "int or float or str", "clustering": "int or float or str", "diameter": None},
        "returns": {"common_neighbors":"List[Int] or List[Float] or List[Str]", "traveling_salesman_problem": "List[Int] or List[Float] or List[Str]","connected_components":"List[Int] or List[Float] or List[Str]", "max_clique": "List[Int] or List[Float] or List[Str]","shortest_path_length": "int", "clustering": "float", "diameter": "int"},
        "directed": False,
        "weighted": True
    },
    "unweighted_directed_tasks": {
        "tasks": ["topological_sort"], # "min_weighted_vertex_cover", "is_bipartite", "hamiltonian_path","strongly_connected_components", "has_eulerian_path", "is_bipartite", "hamiltonian_path","strongly_connected_components", "has_eulerian_path"
        "args": {"min_weighted_vertex_cover": ["graphs"],"is_bipartite": ["graphs"], "hamiltonian_path": ["graphs"], "strongly_connected_components": ["graphs"], "has_eulerian_path": ["graphs"], "topological_sort": None},
        "inputs": {"min_weighted_vertex_cover": None, "is_bipartite": None, "hamiltonian_path": None, "strongly_connected_components": None, "has_eulerian_path": None, "topological_sort": None},
        "returns": {"min_weighted_vertex_cover":"List[Int] or List[Float] or List[Str]", "is_bipartite": "bool", "hamiltonian_path": "List[Int]", "strongly_connected_components": "List[Int]", "has_eulerian_path": "bool", "topological_sort": "List[Int]"},
        "directed": True,
        "weighted": False
    },
    "weighted_directed_tasks": {
        "tasks": [], #   "traveling_salesman_problem", "maximum_flow"
        "args": {"traveling_salesman_problem": ["graphs","nodes"], "maximum_flow":["graphs","_snode", "_tnode"]},
        "inputs": {"traveling_salesman_problem":"int or float or str", "maximum_flow": "int or float or str"},
        "returns": {"traveling_salesman_problem": "List[Int] or List[Float] or List[Str]","maximum_flow": "int or float"},
        "directed": True,
        "weighted": True
    },
    "unweighted_undirected_tasks": {
        "tasks": [], # "topological_sort", "maximum_independent_set", "max_clique", "connected_components", "min_weighted_vertex_cover","common_neighbors", "clustering", "diameter", "is_regular", "is_distance_regular", "is_connected", "has_eulerian_path"
        "args": {"maximum_independent_set":["graphs"],"max_clique": ["graphs"], "connected_components":["graphs"],"min_weighted_vertex_cover": ["graphs"],"common_neighbors": ["graphs","source", "target"],"clustering": ["graphs","nodes"], "topological_sort": ["graphs"], "diameter": ["graphs"], "is_regular": ["graphs"], "is_distance_regular": ["graphs"], "is_connected": ["graphs"], "has_eulerian_path": ["graphs"]},
        "inputs":  {"maximum_independent_set":None,"max_clique": None,"connected_components":None,"min_weighted_vertex_cover": None,"common_neighbors":"int or float or str","clustering": "int or float or str", "topological_sort": None, "diameter": None, "is_regular": None, "is_distance_regular": None, "is_connected": None, "has_eulerian_path": None},
        "returns": {"maximum_independent_set":"List[Int] or List[Float] or List[Str]","max_clique": "List[Int] or List[Float] or List[Str]", "connected_components":"List[Int] or List[Float] or List[Str]","min_weighted_vertex_cover":"List[Int] or List[Float] or List[Str]","common_neighbors":"List[Int] or List[Float] or List[Str]", "clustering": "float", "topological_sort": "List[Tuple[Int]]", "diameter": "int", "is_regular": "bool", "is_distance_regular": "bool", "is_connected": "bool", "has_eulerian_path": "bool"},
        "directed": False,
        "weighted": False
    }
    }

task_description_base_path = pathlib.Path("evaluation_dataset/graphtutor_dataset_scripts/task_descriptions")
output_dir = pathlib.Path("evaluation_dataset/graphtutor_dataset_scripts/task_descriptions")

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
        # Generate graphs for this category
        code_generator = create_code_generator("gpt-5")
        
        task_name_file = f"{task_name}.txt"
        task_description_path = task_description_base_path / is_directed / is_weighted / task_name_file
        with open(task_description_path, "r") as file:
            task_description = file.read()
        seen_real_world_case = f"{task_name}_real_world_case1.txt"
        real_world_task_description_path = task_description_base_path / is_directed / is_weighted / seen_real_world_case
        with open(real_world_task_description_path, "r") as file:
            real_world_task_description = file.read()
        
        query = f"Given a(an) {is_weighted} {is_directed} graph, and the task description:\n{task_description}\n, paraphrase the task description to a real world case scenario different from this {real_world_task_description}."

        result = generate_code_with_openai(
            query=query, 
            openai_model=code_generator, 
            retrieved_docs=None
        )
        print(f"result: {result} ")
        # Save results
        output_path = output_dir / is_directed/ is_weighted / f"{task_name}_real_world_case2.txt"
        with open(output_path, "w") as f:
            f.write(result)
        print(f"    Saved to: {output_path}")