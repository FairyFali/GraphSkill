import json
import re
import os

def read_json_file_names_into_list(folder_path):
    json_file_names = []
    # Iterate over all files in the specified folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a .json file
        if file_name.endswith(".json"):
            key_name = os.path.splitext(file_name)[0]
            json_file_names.append(key_name)
    return json_file_names

def convert_json(old_json_path, new_json_path):
    """
    Converts the old JSON format to the new JSON format:
      {
        "0": {
          "graph": "...",
          "question": "(14, 2)",
          "answer": "The answer is yes.",
          "difficulty": "hard"
        }
      }
    """

    # 1. Read the old JSON file
    with open(old_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = {}

    for key, value in data.items():
        # The field 'question' contains both the graph and the question text:
        # e.g.
        #   "Determine if there is a path ...\nGraph: (0,12) ... (28,30)\nQ: Is there a path between node 14 and node 2?\nA:"
        old_question = value.get('question', '')

        # 2. Extract the graph string using a regex to capture everything
        #    after "Graph:" and before "Q:" (including any line breaks).
        graph_match = re.search(r'Graph:\s*(.*?)\s*Q:', old_question, re.DOTALL)
        if graph_match:
            graph_str = graph_match.group(1).strip()
        else:
            graph_str = ""

        # 3. Extract the two node numbers from the question text:
        #    "Is there a path between node 14 and node 2?"
        #    We'll store the question as "(14, 2)".
        question_match = re.search(r'Is there a path between node (\d+) and node (\d+)\?', old_question)
        if question_match:
            node1, node2 = question_match.groups()
            question_str = f"({node1}, {node2})"
        else:
            question_str = ""

        # 4. Build the new dictionary structure for this key
        new_data[key] = {
            "graph": graph_str,
            "question": question_str,
            "answer": value.get("answer", ""),
            "difficulty": value.get("difficulty", "")
        }

    # 5. Write the new data to a JSON file
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)

if __name__ == "__main__":
    NLgraph_connectivity_path = "../evaluation_dataset/connectivity"
    processed_NLgraph_connectivity_path = "../evaluation_dataset/connectivity/processed"
    json_file_names = read_json_file_names_into_list(NLgraph_connectivity_path)
    for file_name in json_file_names:
        convert_json(f"{NLgraph_connectivity_path}/{file_name}.json", f"{processed_NLgraph_connectivity_path}/processed_{file_name}.json")
