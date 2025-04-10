import os
import numpy as np

PACKAGE_NAME = "rnbrw"

# Create the base directory structure
def create_package_structure():
    # Create main package directory
    os.makedirs(PACKAGE_NAME, exist_ok=True)
    
    # Create the __init__.py file in the main package
    with open(os.path.join(PACKAGE_NAME, "__init__.py"), "w") as f:
        f.write('"""Renewal Non-Backtracking Random Walk (RNBRW) for community detection."""\n\n')
        f.write('from . import weights\n')
        f.write('from . import community\n\n')
        f.write('__version__ = "0.1.0"\n')
    
    # Create subdirectories for package components
    os.makedirs(os.path.join(PACKAGE_NAME, "weights"), exist_ok=True)
    os.makedirs(os.path.join(PACKAGE_NAME, "community"), exist_ok=True)
    os.makedirs(os.path.join(PACKAGE_NAME, "utils"), exist_ok=True)
    
    # Create __init__.py files in subdirectories
    for subdir in ["weights", "community", "utils"]:
        with open(os.path.join(PACKAGE_NAME, subdir, "__init__.py"), "w") as f:
            if subdir == "weights":
                f.write('from .rnbrw import compute_weights\n')
            elif subdir == "community":
                f.write('from .louvain import detect_communities_louvain\n')
    
    print(f"✅ Package structure for {PACKAGE_NAME} created.")

# Create the setup.py file
def create_setup_py():
    setup_content = '''from setuptools import setup, find_packages

setup(
    name="rnbrw",
    version="0.1.0",
    author="Behnaz Moradi-Jamei",
    description="Renewal Non-Backtracking Random Walk (RNBRW) for community detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rnbrw",
    packages=find_packages(),
    install_requires=[
        "numpy", "scipy", "matplotlib", "networkx", "joblib", "python-louvain"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
'''
    with open("setup.py", "w") as f:
        f.write(setup_content)
    print("✅ setup.py file created.")

# Create the README.md file
def create_readme_md():
    readme_content = '''# RNBRW
A Python package to compute Renewal Non-Backtracking Random Walk (RNBRW) edge weights for community detection.

## Features
- Parallel RNBRW edge weight estimation
- Seamless integration with Louvain
- Based on [Moradi-Jamei et al., 2019](https://arxiv.org/abs/1805.07484)

## Installation
```bash
pip install rnbrw
```

## Usage
```python
import networkx as nx
from rnbrw.weights import compute_weights
from rnbrw.community import detect_communities_louvain

# Create or load a graph
G = nx.karate_club_graph()

# Compute RNBRW weights
G = compute_weights(G, nsim=1000, n_jobs=4)

# Detect communities
partition = detect_communities_louvain(G)
```

## Documentation
Full documentation is available at [Read the Docs](https://rnbrw.readthedocs.io).

## Citation
If you use this package in your research, please cite:
```
@article{moradijamei2019renewal,
  title={Renewal Non-Backtracking Random Walks for Community Detection},
  author={Moradi-Jamei, Behnaz and Golnari, Golshan and Zhang, Yanhua and Lagergren, John and Chawla, Nitesh},
  journal={arXiv preprint arXiv:1805.07484},
  year={2019}
}
```

## License
MIT
'''
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("✅ README.md file created.")

# Create the weights implementation file
def create_weights_implementation():
    weights_content = '''"""Implementation of Renewal Non-Backtracking Random Walk (RNBRW) weight computation."""

import numpy as np
import networkx as nx
import time
from joblib import Parallel, delayed
# Import utility functions if needed
# from ..utils.random_walk import rnbrw_simulation

def walk_hole_E(G, seed=None):
    """Perform a non-backtracking random walk with cycle detection.
    
    Parameters
    ----------
    G : networkx.Graph
        The input graph
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    numpy.ndarray
        Array of cycle counts for each edge
    """
    if seed is not None:
        np.random.seed(seed)
    
    m = G.number_of_edges()
    E = list(G.edges())
    T = np.zeros(m, dtype=int)
    L = np.random.choice(m, m, replace=True)
    E_sampled = [E[i] for i in L]
    
    for x, y in E_sampled:
        for u, v in [(x, y), (y, x)]:
            walk = [u, v]
            while True:
                nexts = list(G.neighbors(v))
                try:
                    nexts.remove(u)
                except ValueError:
                    pass
                
                if not nexts:
                    break
                    
                nxt = np.random.choice(nexts)
                if nxt in walk:
                    T[G[v][nxt]['enum']] += 1
                    break
                    
                walk.append(nxt)
                u, v = v, nxt
    
    return T


def compute_weights(G, nsim=1000, n_jobs=1, weight_attr='rnbrw_weight', seed_base=0):
    """Compute RNBRW edge weights for a graph using cycle propagation.
    
    Parameters
    ----------
    G : networkx.Graph
        The input graph
    nsim : int, optional
        Number of random walk simulations, by default 1000
    n_jobs : int, optional
        Number of parallel jobs, by default 1
    weight_attr : str, optional
        Name of the edge attribute to store weights, by default 'rnbrw_weight'
    seed_base : int, optional
        Base random seed, by default 0
    
    Returns
    -------
    networkx.Graph
        The input graph with RNBRW weights added as edge attributes
    
    References
    ----------
    .. [1] Moradi-Jamei, B., Golnari, G., Zhang, Y., Lagergren, J., & Chawla, N. (2019).
           Renewal Non-Backtracking Random Walks for Community Detection.
           arXiv preprint arXiv:1805.07484.
    """
    import time
    from joblib import Parallel, delayed
    
    # Copy the graph to avoid modifying the original
    G_copy = G.copy()
    
    # Start time for performance tracking
    start_time = time.time()
    
    # Initialize edge enumeration
    edges = list(G_copy.edges())
    m = len(edges)
    for i, (u, v) in enumerate(edges):
        G_copy[u][v]["enum"] = i
        G_copy[u][v][weight_attr] = 0.01  # Initialize with small value
    
    # Run parallel simulations
    results = Parallel(n_jobs=n_jobs)(
        delayed(walk_hole_E)(G_copy, seed=seed_base + i) for i in range(nsim)
    )
    
    # Aggregate results from all simulations
    T = sum(results)
    total = T.sum() or 1  # Avoid division by zero
    
    # Update edge weights
    for i, (u, v) in enumerate(edges):
        G_copy[u][v][weight_attr] = T[i] / total if i < len(T) else 0.0
    
    print(f"RNBRW weights computation completed in {time.time() - start_time:.2f} seconds")
    
    return G_copy
'''
    with open(os.path.join(PACKAGE_NAME, "weights", "rnbrw.py"), "w") as f:
        f.write(weights_content)
    print("✅ weights implementation created.")

# Create the community detection implementation file
def create_community_implementation():
    community_content = '''"""Community detection algorithms using RNBRW weights."""

import community as community_louvain
import networkx as nx

def detect_communities_louvain(G, weight_attr='rnbrw_weight', random_state=None):
    """Apply Louvain community detection with RNBRW weights.
    
    Parameters
    ----------
    G : networkx.Graph
        Graph with RNBRW weights (output from compute_weights function)
    weight_attr : str, optional
        Name of the edge attribute containing weights, by default 'rnbrw_weight'
    random_state : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary mapping node to community ID
    
    Notes
    -----
    This function requires the python-louvain package.
    """
    # Check if graph has the required weight attribute
    for u, v in G.edges():
        if weight_attr not in G[u][v]:
            raise ValueError(f"Edge ({u}, {v}) does not have '{weight_attr}' attribute. "
                            "Run compute_weights() first.")
    
    # Apply Louvain algorithm
    partition = community_louvain.best_partition(G, 
                                               weight=weight_attr,
                                               random_state=random_state)
    
    return partition
'''
    with open(os.path.join(PACKAGE_NAME, "community", "louvain.py"), "w") as f:
        f.write(community_content)
    print("✅ community detection implementation created.")

# Create utility functions
def create_utils_implementation():
    utils_content = '''"""Utility functions for RNBRW simulations."""

import random
import networkx as nx
from collections import defaultdict

def rnbrw_simulation(G, max_steps=1000, seed=None):
    """Perform a single RNBRW simulation and return edge traversal counts.
    
    Note: This is the older implementation. The package now uses the more
    efficient walk_hole_E function for simulations.
    
    Parameters
    ----------
    G : networkx.Graph
        The input graph
    max_steps : int, optional
        Maximum number of steps in the random walk, by default 1000
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary mapping edges (u, v) to their traversal counts
    """
    if seed is not None:
        random.seed(seed)
    
    # Initialize edge counts
    edge_counts = defaultdict(int)
    
    # Choose a random starting node
    current_node = random.choice(list(G.nodes()))
    prev_node = None
    
    # Walk path
    walk = [current_node]
    
    # Perform random walk
    steps = 0
    while steps < max_steps:
        steps += 1
        
        # Get neighbors excluding the previous node (non-backtracking)
        neighbors = list(G.neighbors(current_node))
        if prev_node is not None and prev_node in neighbors:
            neighbors.remove(prev_node)
        
        # If no valid neighbors, break
        if not neighbors:
            break
        
        # Choose next node randomly from valid neighbors
        next_node = random.choice(neighbors)
        
        # Check if the next node creates a cycle
        if next_node in walk:
            # Record the edge where cycle occurs
            edge = tuple(sorted([current_node, next_node]))
            edge_counts[edge] += 1
            break
        
        # Add to walk and move to next node
        walk.append(next_node)
        prev_node = current_node
        current_node = next_node
    
    return edge_counts
'''
    with open(os.path.join(PACKAGE_NAME, "utils", "random_walk.py"), "w") as f:
        f.write(utils_content)
    print("✅ utility functions created.")

# Create an example script
def create_example_script():
    example_content = '''"""Example usage of the RNBRW package."""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 
import time
from rnbrw.weights import compute_weights
from rnbrw.community import detect_communities_louvain

def main():
    """Run RNBRW community detection on the Zachary's Karate Club graph using cycle propagation."""
    # Create a graph
    print("Creating Zachary's Karate Club graph...")
    G = nx.karate_club_graph()
    
    # Compute RNBRW weights
    print("Computing RNBRW weights using cycle propagation (this may take a moment)...")
    G = compute_weights(G, nsim=1000, n_jobs=4, seed_base=42)
    
    # Detect communities
    print("Detecting communities using Louvain method...")
    partition = detect_communities_louvain(G)
    
    # Print results
    print(f"Found {len(set(partition.values()))} communities")
    for community_id in sorted(set(partition.values())):
        nodes = [node for node, comm in partition.items() if comm == community_id]
        print(f"Community {community_id}: {len(nodes)} nodes")
    
    # Visualize the results
    print("Visualizing results...")
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes based on community
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    node_colors = [colors[partition[node] % len(colors)] for node in G.nodes()]
    
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title("RNBRW Community Detection on Zachary's Karate Club")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('rnbrw_communities.png')
    plt.show()
    
    print(f"Results saved to 'rnbrw_communities.png'")

if __name__ == "__main__":
    main()
'''
    with open("example.py", "w") as f:
        f.write(example_content)
    print("✅ example script created.")

# Main function to create all files
def create_rnbrw_package():
    create_package_structure()
    create_setup_py()
    create_readme_md()
    create_weights_implementation()
    create_community_implementation()
    create_utils_implementation()
    create_example_script()
    print("\n✨ RNBRW package created successfully! ✨")

if __name__ == "__main__":
    create_rnbrw_package()