import random
import os

def generate_random_weighted_graph(n, e, filename='graph_weighted.txt', weight_range=(1, 10)):
    """
    Generate an undirected, weighted graph with n nodes and e edges, 
    and store the edge list in a text file.
    
    Each line of the output file will have the format:
      u v w
    where u and v are node labels (1-based), and w is the weight of the edge.
    
    Parameters:
    -----------
    n : int
        Number of nodes in the graph.
    e : int
        Number of edges to include in the graph.
    filename : str
        File name to save the edge list.
    weight_range : tuple
        A 2-element tuple (low, high) specifying the (inclusive) range
        of edge weights to pick from.
    """
    # A simple check to avoid attempting to create too many edges
    # for a simple undirected graph without self-loops
    max_edges = n * (n - 1) // 2
    if e > max_edges:
        raise ValueError(
            f"Number of edges e = {e} exceeds "
            f"maximum possible edges = {max_edges} for {n} nodes."
        )
    
    edges = set()
    
    # Keep adding random edges until we have exactly e edges
    while len(edges) < e:
        u = random.randint(1, n)
        v = random.randint(1, n)
        
        if u != v:
            # For an undirected graph, store edge as a sorted tuple (smallest first)
            edge = tuple(sorted((u, v)))
            edges.add(edge)
    
    # Write the edge list, including a random weight for each edge
    with open(filename, 'w') as f:
        for (u, v) in edges:
            w = random.randint(weight_range[0], weight_range[1])
            f.write(f"{u} {v} {w}\n")

if __name__ == "__main__":
    # Example usage:
    # Generate a weighted graph of 75 nodes, 2000 edges, 
    # with weights between 1 and 10, stored in 'graph_n75_e2000_weighted.txt'
    n = 100
    e = 4000
    graph_file_name = f'graph_n{n}_e{e}_weighted.txt'
    graph_path = os.path.join("../../evaluation_dataset/graphs", "weighted_graphs", graph_file_name)
    generate_random_weighted_graph(n, e, graph_path, weight_range=(1, 10))
    print(f"Generated a weighted graph with {n} nodes and {e} edges. See '{graph_file_name}' for the edge list.")

