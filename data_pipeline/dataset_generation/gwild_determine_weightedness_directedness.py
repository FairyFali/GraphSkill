import json
import pathlib, sys;sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from pathlib import Path
from utils.get_llm_response_generator import create_code_generator
from utils.generation_functions.generate_prompt import generate_code_with_openai

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

graph_definitions = {
    "weighted_graph": (
        "A graph in which every edge is associated with a numerical value (weight) "
        "that typically represents cost, distance, capacity, or some other metric."
    ),
    "unweighted_graph": (
        "A graph whose edges carry no explicit weights; all edges are implicitly "
        "treated as equal (often as weight = 1) when performing algorithms."
    ),
    "directed_graph": (
        "A graph whose edges have an orientation, meaning each edge points from "
        "one vertex (tail) to another (head); edge (u,v) is distinct from (v,u)."
    ),
    "undirected_graph": (
        "A graph whose edges do not have orientation; an edge {u,v} can be "
        "traversed in either direction, and the pair (u,v) is identical to (v,u)."
    ),
}

print("Initialising code generate agent")
openai_model_code_generate = create_code_generator(
    model_name="deepseek-reasoner",
    system_prompt="You are an expert in graph theory and graph-related tasks"
    )
print("DeepSeek Code Generation Agent Initialised")

example_description_path = f"evaluation_dataset//GWild_docs/sampled_tasks/sampled_tasks.json"
with open(example_description_path, "r", encoding="utf-8") as f:
    sampled_tasks = json.load(f)
task_path = f"/data/chenglin/GraphTutor/Dataset/training_dataset/gwild_filtered.json"
with open(task_path, "r", encoding="utf-8") as f:
    tasks = json.load(f)
task_name_graph_type_dict = {}
for _, task_nx_example_list in sampled_tasks.items():
    for task_nx_example in task_nx_example_list:
        task_name = task_nx_example["function"]
        # task_doc = task_nx_example[""]
        question = tasks[task_name][0]["input"]
        graph_def = json.dumps(graph_definitions)
        # import pdb;pdb.set_trace()
        user_query = (
            f"Based on the input graph task {task_name}",
            f"Carefully look at the graph categories and explanation in this dictionary: {graph_def}",
            f"Here is a sample question for the given graph task {question}",
            f"Select the graph categories that can be used in the input graph task from this given dictionary. And return the selected graph categories in a list"
            f"Return your answer without any explanation, "
            )
        result = generate_code_with_openai(
        query=json.dumps(user_query), 
        openai_model=openai_model_code_generate, 
        retrieved_docs=None,
        instrcution="You are a sophisticated AI expert in graph theory and algorithms.")
        task_name_graph_type_dict[task_name] = result
        save_dict_to_json(task_name_graph_type_dict, f"evaluation_dataset/GWild_docs/graph_weightedness_directedness_for_task/weightedness_directedness.json")
