import json
import os
import glob
from file_processing_helpers.find_folders import get_folders_in_directory
from file_processing_helpers.find_files import find_question_files, check_file_existence


def merge_question_files(question_files, output_file):
    """
    Scans 'folder_path' for all JSON files ending with '_{task_name}_questions.json',
    merges their contents into a single JSON array, and writes to 'output_file'.
    """
    merged_data = []
    
    # Use glob to find all matching JSON files in the specified folder
    for question_path in question_files:
        file_idx = question_path.split("/")[-1].split("_")[5]
        with open(question_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        data["index"] = file_idx
        merged_data.append(data)

    # Write the merged data to the output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(merged_data, outfile, indent=4)

# Example usage:
if __name__ == "__main__":
    task_name = "shortest_path"
    base_folder = f"/home/cvw5844/exp_code/LLM_graph_task/evaluation_dataset/graphs/{task_name}"
    graph_type_folders = get_folders_in_directory(base_folder)
    file_pattern = f"_{task_name}_questions.json"
    

    for graph_type_folder in graph_type_folders:
        graph_type_base_folder = os.path.join(base_folder, graph_type_folder) #has graph type "directed" "undirected"
        graph_size_folders = get_folders_in_directory(graph_type_base_folder)
        for graph_size_folder in graph_size_folders:
            question_base_folder = os.path.join(graph_type_base_folder, graph_size_folder)
            questionfiles = find_question_files(question_base_folder, task_name=task_name)
            all_file_exists = check_file_existence(questionfiles)
            if all_file_exists:
                output_file = base_folder + f"/{graph_type_folder}/{graph_size_folder}/merged_varied_graph_{task_name}_{graph_type_folder}_{graph_size_folder}.json"
                merge_question_files(question_files=questionfiles, output_file=output_file)
            else: 
                print(f"There are files that cannot be found, please examin your base_folder")
