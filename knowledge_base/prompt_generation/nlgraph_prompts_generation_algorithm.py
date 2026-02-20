import jsonlines
import json
import random as rd
import ast
import os
from utils.get_llm_response_generator import create_code_generator
from utils.generation_functions.retrieve_doc_chapter import retrieve_documentation_chapter
from huggingface_hub import login
from knowledge_base.doc_retrieval.retrieve_nx_documentation_llm import retrieve_doc
from utils.generation_functions.generate_prompt import generate_code_with_openai
from pathlib import Path
import re, unicodedata

def return_query(task_description):
    return f"""
Input to you consists of a single block:

<TASK>
{task_description}
</TASK>

Your job is to output **one finished user prompt** (nothing else) that follows ALL rules below.

–––––  RULES  –––––
0. **Choose the best-suited algorithm or mathematical formula** for solving the task you see inside <TASK> … </TASK>.  
1. **Begin** with a concise (2 – 4 sentences) plain-English summary of that algorithm/formula.  
2. In **≤ 1 sentence**, state *why* this algorithm is appropriate for the task.  
3. **Generate five original graph examples** that are **structurally different** from the graph in <TASK>.  
   • For each example, specify the node set and list the edges as unordered pairs, e.g. `(4, 2) (0, 4) …`.  
   • Vary the number of nodes (within a sensible range) and edge patterns across the five examples.  
4. For **each** of the five examples:  
   a. Restate the edge list.  
   b. Ask the question exactly as in <TASK> but with the new graph.  
   c. Provide a step-by-step chain-of-thought solution that explicitly walks through the algorithm you chose, ending with the final answer.  
5. After the five worked examples, **quote the original <TASK> description verbatim** and label it **“Target Task”.**  
6. Tell the model to solve the Target Task **using the same algorithm** and to **show its reasoning**.  
7. End with: Provide the final answer clearly labelled. 

Return **only** the complete, ready-to-use user prompt that satisfies all rules. Do **not** output any extra commentary or metadata."""

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

def find_graph_jsonl_files(base_folder):
    """
    Recursively walks through the given folder and returns a list of files
    that end with '_{task_name}_quetions.json'.
    """
    question_files = []
    for root, dirs, files in os.walk(base_folder):
        for file_name in files:
            if file_name.endswith(f'.jsonl'):
                # If you only want to print them, you could do:
                # print(os.path.join(root, file_name))

                # Here we'll collect them in a list:
                question_files.append(os.path.join(root, file_name))
                # info_files.append(os.path.join(root, root.split('/')[-1]+".json")) # extract the dataset name from '{dataset_name}_shortest_path_questions.json' file
    return question_files

def read_txt_or_none(filepath: str | Path, encoding: str = "utf-8") -> str | None:
    """
    Read a text file and return its contents.
    If the file does not exist (or isn’t a regular file), return None.

    Parameters
    ----------
    filepath : str | Path
        Path to the .txt file.
    encoding : str, optional
        Text encoding to use when opening the file (default: "utf-8").

    Returns
    -------
    str | None
        The file contents as a single string, or None if the path is missing.
    """
    path = Path(filepath)
    if not path.is_file():          # file doesn’t exist or isn’t a regular file
        return None
    try:
        return path.read_text(encoding=encoding)
    except OSError:                 # covers permission errors, etc.
        return None

def main():
        # 5. Initialize the OpenAI generator with your API key
    print("Initialising code generate agent")
    openai_model_code_generate = create_code_generator(
        model_name="deepseek-chat",
        system_prompt="You are an expert prompt engineer."
        )
    print("DeepSeek Code Generation Agent Initialised")
    example_description_path = f"evaluation_dataset/GWild/function_pages/task_examples.json"
    with open(example_description_path, "r", encoding="utf-8") as f:
        task_nx_examples = json.load(f)
    task_path = f"/data/chenglin/GraphTutor/Dataset/training_dataset/gwild_filtered.json"
    with open(task_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    generated_programs = {}
    for task_nx_example in task_nx_examples[:100]:
        task_name = task_nx_example["function"]
        task_desc = tasks[task_name][0]["input"]
        input_query = return_query(task_desc)
        result = generate_code_with_openai(
            query=input_query, 
            openai_model=openai_model_code_generate, 
            retrieved_docs=None)
        print("\n[Generated Code]\n", result)
        generated_programs[task_name] = result
        print(f"Save program: {result} to path evaluation_dataset/GWild/NLGraph_prompt/Algorithm-CoT.json")
        save_dict_to_json(generated_programs, f"evaluation_dataset/GWild/NLGraph_prompt/Algorithm-CoT.json")
    del openai_model_code_generate        

if __name__ == "__main__":
    main()
