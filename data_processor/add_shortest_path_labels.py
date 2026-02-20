import os
from tqdm import tqdm
import gc
import json
import gzip
import csv
import heapq
import random
from collections import defaultdict, deque

def load_graph_gz(filename, directed=True, weighted=False):
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
            if not weighted:
                for row in reader:
                    if len(row) < 2:
                        continue
                    u, v = map(int, row)  # Convert node labels to integers
                    graph[u].add(v)
                    if not directed:
                        # For undirected graphs, add the reverse edge
                        graph[v].add(u)
            else:
                for row in reader:
                    if len(row) < 2:
                        continue
                    u = int(row[0])
                    v = int(row[1])
                    w = int(row[2])  # Convert node labels to integers
                    graph[u].add((v,w))
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

def bfs_shortest_path(graph, start, end):
    """
    Find a shortest path in an unweighted graph from `start` to `end` using BFS.
    
    :param graph: A dict mapping node -> set/list of neighbors.
    :param start: The starting node.
    :param end: The target node.
    :return: A list of nodes representing the shortest path from start to end, or None if no path exists.
    """
    if start == end:
        return [start]
    
    visited = set([start])
    queue = deque([start])
    
    # Keep track of how we reached each node (predecessor)
    predecessor = {start: None}
    
    while queue:
        current = queue.popleft()
        
        # Early exit if we reach the end
        if current == end:
            break
        
        # Add neighbors
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                predecessor[neighbor] = current
                queue.append(neighbor)
    
    # If we never reached `end`, then no path exists
    if end not in visited:
        return None
    
    # Reconstruct the path from end to start by following predecessors
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = predecessor[node]
    
    # The path is built backwards so we reverse it
    path.reverse()
    return path


def dijkstra_shortest_path(graph, start, end):
    """
    Find the shortest path in a weighted directed graph from 'start' to 'end' using Dijkstra's Algorithm.
    
    :param graph: Dict mapping node -> list of (neighbor, weight).
    :param start: The starting node.
    :param end:   The target node.
    :return: (distance, path) where:
                - distance is the minimum cost (float('inf') if unreachable),
                - path is a list of nodes from start to end (empty if unreachable).
    """
    # Distances dictionary: cost to reach each node from 'start'
    distances = {node: float('inf') for node in graph}
    distances[start] = 0.0
    
    # Predecessor map to reconstruct paths
    predecessor = {node: None for node in graph}
    
    # Min-heap / priority queue for the nodes to explore (distance_so_far, node)
    pq = [(0.0, start)]
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        # If this node is the end node, we can stop early
        if current_node == end:
            break
        
        # If we already have a better distance for this node, skip
        if current_dist > distances[current_node]:
            continue
        
        # Explore neighbors
        for neighbor, weight in graph[current_node]:
            distance_via_current = current_dist + weight
            if distance_via_current < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance_via_current
                predecessor[neighbor] = current_node
                heapq.heappush(pq, (distance_via_current, neighbor))
    
    # If 'end' was never updated, it's unreachable
    if distances.get(end, float('inf')) == float('inf'):
        return float('inf'), []
    
    # Reconstruct the path from end to start using predecessor
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = predecessor[node]
    path.reverse()
    
    return distances[end], path

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

def find_csv_gz_files(base_folder):
    """
    Recursively walks through the given folder and returns a list of files
    that do NOT end with '.txt.gz'.
    """
    csv_gz_files = []
    
    for root, dirs, files in os.walk(base_folder):
        for file_name in files:
            if file_name.endswith('.csv.gz'):
                # If you only want to print them, you could do:
                # print(os.path.join(root, file_name))
                
                # Here we'll collect them in a list:
                csv_gz_files.append(os.path.join(root, file_name))
    
    return csv_gz_files

def main(file_path: str, file_name: str, directed: bool, weighted: bool):
    # Replace 'graph.csv.gz' with the actual file path to your .txt.gz or .csv.gz
    # Set directed=False for undirected graphs; True for directe
    original_filename = file_name.split(".")[0]
    labels_file_name = f"{original_filename}_shortest_path_questions.json"
    # 1. Load the graph
    if not os.path.exists(os.path.join(file_path, labels_file_name)):
        try:
            filename = os.path.join(file_path, file_name)
            graph = load_graph_gz(filename, directed=directed, weighted=weighted)
            
            # 2. Randomly select one node
            all_nodes = list(graph.keys())
            random_node = random.choice(all_nodes)
            
            # 3. Find all connected (reachable) nodes from the randomly selected node
            connected = list(bfs(graph, random_node))
            target_node=random.choice(connected)
            if weighted:
                weight, path = dijkstra_shortest_path(graph, random_node, target_node)
            else:
                path = bfs_shortest_path(graph, random_node, target_node)
            
            if directed is True:
                shortest_path_question = f'In a directed graph. Note that (i,j) means that node i and node j are connected with an directed edge from i to j.'
                if weighted is True:
                    shortest_path_question = f'In a weighted directed graph. Note that (i,j,w) means that node i and node j are connected with an directed edge from i to j with the weight w.'
            elif directed is False:
                shortest_path_question = f'In an undirected graph. Note that (i,j) means that node i and node j are connected with an undirected edge.'
            else:
                print(f"The value directed is not provided")
            
            if len(connected) > 0 and not weighted:
                data = {
                    "file_name": filename,
                    "shortest_path_question": f"{shortest_path_question} Q: What is the shortest path between node {str(random_node)} and node {str(target_node)}?",
                    "answer": f"The shortest path from node {str(random_node)} to node {str(target_node)} is {path}",
                        }
                with open(os.path.join(file_path, labels_file_name), "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=4)
            elif len(connected) > 0 and weighted and weight != float('inf'):
                data = {
                    "file_name": filename,
                    "shortest_path_question": f"{shortest_path_question} Q: What is the shortest path between node {str(random_node)} and node {str(target_node)}?",
                    "answer": f"The shortest path from node {str(random_node)} to node {str(target_node)} is {path}, with a total weight of {weight}",
                        }
                with open(os.path.join(file_path, labels_file_name), "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=4)
            else:
                print(f"NNo shortest path found for {file_path}.")

            print(f"Dictionary saved to {os.path.join(file_path, labels_file_name)}")
            # 4. Print results
            print(f"Randomly selected node: {random_node}")
            print(f'Number of connected nodes {len(connected)}')
            print(f'Number of all nodes {len(all_nodes)}')
            gc.collect()
        except:
            print(f"Unsupported file {file_name}")
    else:
        print(f"{labels_file_name} exists, skip generating coonectivity questions")

if __name__ == "__main__":
    base_folder = "/home/cvw5844/dataset/large_graph_dataset"
    txt_graph_files = find_txt_gz_files(base_folder)
    csv_graph_files = find_csv_gz_files(base_folder)
    graph_files = txt_graph_files+csv_graph_files
    for graph_file in tqdm(graph_files):
        if os.path.getsize(graph_file) > 85470044:
            print(f'The file {graph_file} is too large for generating shortest path questions')
            continue
        root_file_path = os.path.join(base_folder, graph_file.split("/")[5], graph_file.split("/")[6])
        file_name = graph_file.split("/")[-1]
        if graph_file.split("/")[5] == "directed":
            main(root_file_path, file_name, directed=True, weighted=False)
        elif graph_file.split("/")[5] == "undirected":
            main(root_file_path, file_name, directed=False, weighted=False)
        elif graph_file.split("/")[5] == "weighted&directed":
            main(root_file_path, file_name, directed=True, weighted=True)
        else:
            continue
