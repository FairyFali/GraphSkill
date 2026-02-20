import json
import os
import glob

def merge_connectivity_files(folder_path, output_file, file_pattern):
    """
    Scans 'folder_path' for all JSON files ending with '_connectivity_questions.json',
    merges their contents into a single JSON array, and writes to 'output_file'.
    """
    merged_data = []
    
    # Use glob to find all matching JSON files in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(file_pattern):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as infile:
                    data = json.load(infile)
                merged_data.append(data)

    # Write the merged data to the output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(merged_data, outfile, indent=4)

# Example usage:
if __name__ == "__main__":
    task_name = "shortest_path"
    folder_path = "/home/cvw5844/exp_code/LLM_graph_task/evaluation_dataset/graphs/"
    output_file = f"../evaluation_dataset/{task_name}/merged_varied_graph_{task_name}.json"
    file_pattern = f"_{task_name}_questions.json"
    merge_connectivity_files(folder_path, output_file, file_pattern)
    print(f"Merged JSON written to {output_file}")
