import os
from tqdm import tqdm
import gc
import json
import gzip
import csv
import heapq
import random
from collections import defaultdict, deque
import networkx as nx
import re

def load_graph_from_edge_list(file_path, directed=True, weighted=None):
    if directed:
        G = nx.read_edgelist(file_path, nodetype=int, create_using=nx.DiGraph())  # Create a directed graph
    else:
        G = nx.read_edgelist(file_path, nodetype=int, create_using=nx.Graph())  # Assuming nodes are integers
    return G

# Compute the shortest path between two arbitrary nodes
def find_shortest_path(graph: nx.graph, source, target):
    try:
        # Compute the shortest path using Dijkstra's algorithm
        path = nx.shortest_path(graph, source=source, target=target, weight=None)
        return path
    except nx.NetworkXNoPath:
        print(f"No path exists between {source} and {target}")
        return None

def find_arbitrary_connected_nodes(graph: nx.graph):
    found_connected_nodes = False
    while not found_connected_nodes:
        source = random.choice(list(graph.nodes))
        target = random.choice(list(graph.nodes))
        if source != target and nx.has_path(graph, source, target):
            found_connected_nodes = True
            return source, target
        
def find_pairs_with_distance_K(G: nx.graph, k: int, max_pairs: int):
    """
    Return all unordered node pairs (u, v) in graph G 
    such that the shortest path distance between u and v is exactly 3.
    """
    pairs = set()
    count = 0
    for source in G.nodes():
        # Get distances from `source` to all other reachable nodes
        distances = nx.single_source_shortest_path_length(G, source, cutoff=k)
        
        # For each node reachable from `source`, check if distance == 3
        for target, dist in distances.items():
            if dist == k:
                # Sort the pair so (u, v) == (v, u), avoids duplicates in an undirected graph
                count += 1
                pair = tuple(sorted((source, target)))
                pairs.add(pair)
            if count >= max_pairs:
                break

    return list(pairs)

def extract_node_and_edge_num(filename_str):
    matches = re.findall(r"\d+", filename_str)
    n, e = map(int, matches)
    return n, e

def find_txt_files(base_folder):
    """
    Recursively walks through the given folder and returns a list of files
    that do NOT end with '.txt.gz'.
    """
    txt_files = []
    
    for root, dirs, files in os.walk(base_folder):
        for file_name in files:
            if file_name.endswith('.txt'):
                # If you only want to print them, you could do:
                # print(os.path.join(root, file_name))
                node, edge = extract_node_and_edge_num(filename_str=file_name)
                if node <= 100 and node >= 50:
                # Here we'll collect them in a list:
                    txt_files.append(os.path.join(root, file_name))
    
    return txt_files

def main(file_path: str, file_name: str, directed: bool, weighted: bool):
    # Replace 'graph.csv.gz' with the actual file path to your .txt.gz or .csv.gz
    # Set directed=False for undirected graphs; True for directe
    filename = os.path.join(file_path, file_name)
    print(f"Loading Graph {filename}")
    graph = load_graph_from_edge_list(filename, directed=directed)
    node_num, edge_num = extract_node_and_edge_num(filename.split("/")[-1])
    # 2. Randomly select one node
    print("Finding Connnected Node Pair")
    #uncomment it for totally random target and source
    # source, target = find_arbitrary_connected_nodes(graph)
    # path = find_shortest_path(graph, source, target)
    pairs_with_dis_k = find_pairs_with_distance_K(graph, k=2, max_pairs=20)

    for i, (source,target) in enumerate(pairs_with_dis_k):
        path = find_shortest_path(graph, source, target)
        original_filename = file_name.split(".")[0]
        filename_with_idx = original_filename + "_s" + str(source) + "_t" + str(target) + "_" + str(i)
        labels_file_name = f"{filename_with_idx}_shortest_path_questions.json"
        if directed:
            graph_type_str = "directed"
            question_file_path = os.path.join(file_path, "shortest_path", "directed")
        else:
            graph_type_str = "undirected"
            question_file_path = os.path.join(file_path, "shortest_path", "undirected")
        root_question_file_path = os.path.join(question_file_path, original_filename)
        os.makedirs(root_question_file_path, exist_ok=True)
        question_file_path = os.path.join(root_question_file_path, labels_file_name)
        # 1. Load the graph
        if not os.path.exists(question_file_path):
            # try:
                
                if directed is True:
                    shortest_path_question = f'In a directed graph. Note that (i,j) means that node i and node j are connected with an directed edge from i to j.'
                    if weighted is True:
                        shortest_path_question = f'In a weighted directed graph. Note that (i,j,w) means that node i and node j are connected with an directed edge from i to j with the weight w.'
                elif directed is False:
                    shortest_path_question = f'In an undirected graph. Note that (i,j) means that node i and node j are connected with an undirected edge.'
                else:
                    print(f"The value directed is not provided")
                
                if not weighted:
                    data = {
                        "file_name": filename,
                        "shortest_path_question": f"{shortest_path_question} Q: What is the shortest path between node {str(source)} and node {str(target)}?",
                        "answer": f"The shortest path from node {str(source)} to node {str(target)} is {path}",
                        "graph_type": graph_type_str,
                        "node_num": node_num,
                        "edge_num": edge_num,
                            }
                    with open(question_file_path, "w", encoding="utf-8") as file:
                        json.dump(data, file, indent=4)
                    print(f"Save graph in {filename} to {question_file_path}")
                elif weighted:
                    print(f'weighted graph not supported')
                    # data = {
                    #     "file_name": filename,
                    #     "shortest_path_question": f"{shortest_path_question} Q: What is the shortest path between node {str(source)} and node {str(target)}?",
                    #     "answer": f"The shortest path from node {str(source)} to node {str(target)} is {path}, with a total weight of {weight}",
                    #         }
                    # with open(os.path.join(file_path, labels_file_name), "w", encoding="utf-8") as file:
                    #     json.dump(data, file, indent=4)
                else:
                    print(f"No shortest path found for {file_path}.")

                print(f"Dictionary saved to {question_file_path}")
                # 4. Print results
                print(f"Randomly selected node: {source} and {target}")
                gc.collect()
        #     except:
    #         print(f"Unsupported file {file_name}")
    # else:
    #     print(f"{labels_file_name} exists, skip generating coonectivity questions")

if __name__ == "__main__":
    base_folder = "/home/cvw5844/exp_code/LLM_graph_task/evaluation_dataset/graphs/"
    txt_graph_files = find_txt_files(base_folder)
    graph_files = txt_graph_files
    for graph_file in tqdm(graph_files):
        directed = True
        file_name = graph_file.split("/")[-1]
        main(base_folder, file_name, directed=directed, weighted=False)
        directed = False
        main(base_folder, file_name, directed=directed, weighted=False)        
        

