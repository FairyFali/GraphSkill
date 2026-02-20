import os
from tqdm import tqdm
import gc
import json
import gzip
import csv
import random
from collections import defaultdict, deque

def load_graph_gz(filename, directed=True):
    """
    Reads a .txt.gz or .csv.gz file containing edges.
    Each line should have at least two entries (e.g., 'u,v').
    Builds and returns an adjacency list as a dict of sets.
    """
    graph = defaultdict(set)
    
    if filename.endswith("csv.gz"):
    # Open the gzipped file in text mode
        with gzip.open(filename, 'rt') as f:
            # Use csv.reader to handle CSV or simple comma-delimited lines
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                u, v = map(int, row)  # Convert node labels to integers
                graph[u].add(v)
                if not directed:
                    # For undirected graphs, add the reverse edge
                    graph[v].add(u)
    elif filename.endswith("txt.gz"):
        with gzip.open(filename, 'rt') as f:
            for line in f:
                # Strip line and split by commas or whitespace
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # import pdb; pdb.set_trace()

                if ',' in line:
                    u_str, v_str = line.split(',')
                else:
                    # fallback: split by whitespace
                    u_str, v_str = line.split()
                
                u, v = int(u_str), int(v_str)
                
                graph[u].add(v)
                if not directed:
                    # For undirected graphs, add the reverse edge
                    graph[v].add(u)
    else:
        print(f"Unsupported file {filename}")


    return graph

def bfs(graph, start_node):
    """
    Performs BFS from the start_node and returns
    all nodes reachable from start_node.
    """
    visited = set([start_node])
    queue = deque([start_node])
    
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited

def find_txt_gz_files(base_folder):
    """
    Recursively walks through the given folder and returns a list of files
    that do NOT end with '.txt.gz'.
    """
    non_txt_gz_files = []
    
    for root, dirs, files in os.walk(base_folder):
        for file_name in files:
            if file_name.endswith('.txt.gz'):
                # If you only want to print them, you could do:
                # print(os.path.join(root, file_name))
                
                # Here we'll collect them in a list:
                non_txt_gz_files.append(os.path.join(root, file_name))
    
    return non_txt_gz_files

def main(file_path: str, file_name: str, directed: bool):
    # Replace 'graph.csv.gz' with the actual file path to your .txt.gz or .csv.gz
    # Set directed=False for undirected graphs; True for directe
    original_filename = file_name.split(".")[0]
    connectivity_labels_file_name = f"{original_filename}_connectivity_questions.json"
    # 1. Load the graph
    if not os.path.exists(os.path.join(file_path, connectivity_labels_file_name)):
        try:
            filename = os.path.join(file_path, file_name)
            graph = load_graph_gz(filename, directed=directed)
            
            # 2. Randomly select one node
            all_nodes = list(graph.keys())
            random_node = random.choice(all_nodes)
            
            # 3. Find all connected (reachable) nodes from the randomly selected node
            connected = list(bfs(graph, random_node))
            disconnected = [x for x in all_nodes if x not in connected]
            
            if directed is True:
                connectivity_question = f'Determine if there is a path between two nodes in the graph. Note that (i,j) means that node i and node j are connected with an directed edge from i to j.'
            elif directed is False:
                connectivity_question = f'Determine if there is a path between two nodes in the graph. Note that (i,j) means that node i and node j are connected with an undirected edge.'
            else:
                print(f"The value directed is not provided")
            
            if len(disconnected) > 0 and len(connected) > 0:
                connected_node = random.choice(connected)
                disconnected_node = random.choice(disconnected)
                data = {
                    "file_name": filename,
                    "connectivity_question": f"{connectivity_question} Q: Is there a path between node {str(random_node)} and node {str(connected_node)}?",
                    "connectivity_answer": "The answer is yes.",
                    "disconnectivity_question": f"{connectivity_question} Q: Is there a path between node {str(random_node)} and node {str(disconnected_node)}?",
                    "disconnectivity_answer": "The answer is no.",
                        }
            elif len(disconnected) > 0 and len(connected) == 0:
                disconnected_node = random.choice(disconnected)
                data = {
                    "file_name": filename,
                    "disconnectivity_question": f"{connectivity_question} Q: Is there a path between node {str(random_node)} and node {str(disconnected_node)}?",
                    "disconnectivity_answer": "The answer is no.",
                        }
            else:
                connected_node = random.choice(connected)
                data = {
                    "file_name": filename,
                    "connectivity_question": f"{connectivity_question} Q: Is there a path between node {str(random_node)} and node {str(connected_node)}?",
                    "connectivity_answer": "The answer is yes.",
                    }
            with open(os.path.join(file_path, connectivity_labels_file_name), "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)

            print(f"Dictionary saved to {os.path.join(file_path, connectivity_labels_file_name)}")
            # 4. Print results
            print(f"Randomly selected node: {random_node}")
            print(f'Number of connected nodes {len(connected)}')
            print(f'Number of all nodes {len(all_nodes)}')
            gc.collect()
        except:
            print(f"Unsupported file {file_name}")
    else:
        print(f"{connectivity_labels_file_name} exists, skip generating coonectivity questions")

if __name__ == "__main__":
    base_folder = "/home/cvw5844/dataset/large_graph_dataset"
    graph_files = find_txt_gz_files(base_folder)
    for graph_file in tqdm(graph_files):
        if os.path.getsize(graph_file) > 85470044:
            print(f'The file {graph_file} is too large for generating connectivity questions')
            continue
        root_file_path = os.path.join(base_folder, graph_file.split("/")[5], graph_file.split("/")[6])
        file_name = graph_file.split("/")[-1]
        if graph_file.split("/")[5] == "directed":
            main(root_file_path, file_name, directed=True)
        elif graph_file.split("/")[5] == "undirected":
            main(root_file_path, file_name, directed=False)
        else:
            continue
