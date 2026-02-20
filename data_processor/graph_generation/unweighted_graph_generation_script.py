import random
import os


def generate_random_graph(n, e, filename='graph.txt'):
    """
    Generate an undirected graph with n nodes and e edges, 
    and store the edge list in a text file.
    
    Parameters:
    -----------
    n : int
        Number of nodes in the graph.
    e : int
        Number of edges to include in the graph.
    filename : str
        File name to save the edge list.
    """
    # A simple check to avoid attempting to create too many edges
    # for a simple undirected graph without self-loops
    max_edges = n * (n - 1) // 2
    if e > max_edges:
        raise ValueError(f"Number of edges e = {e} exceeds maximum possible edges = {max_edges} for {n} nodes.")
    
    edges = set()
    
    # Keep adding random edges until we have exactly e edges
    while len(edges) < e:
        # Pick two distinct nodes
        u = random.randint(1, n)
        v = random.randint(1, n)
        
        if u != v:
            # For an undirected graph, store edge as a sorted tuple (smallest first)
            edge = tuple(sorted((u, v)))
            edges.add(edge)
    
    # Write the edge list to the output file
    with open(filename, 'w') as f:
        for (u, v) in edges:
            f.write(f"{u} {v}\n")

if __name__ == "__main__":
    # Example usage:
    # Generate a graph of 10 nodes, 15 edges, and store in "graph.txt"
    n = 75
    e = 2000
    graph_file_name = f'graph_n{n}_e{e}.txt'
    graph_path = os.path.join("../../evaluation_dataset/graphs", graph_file_name)
    generate_random_graph(n, e, graph_path)
    print(f"Generated a graph with {n} nodes and {e} edges. See 'graph.txt' for the edge list.")
