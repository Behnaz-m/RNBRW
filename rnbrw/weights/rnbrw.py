"""Implementation of Renewal Non-Backtracking Random Walk (RNBRW) weight computation."""

import numpy as np
import networkx as nx
import time
from joblib import Parallel, delayed
from rnbrw.utils import normalize_edge_weights

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
    L = np.random.choice(m, m, replace=False)
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
    # Normalize them to sum to 1
    normalize_edge_weights(G_copy, weight=weight_attr)
    print(f"RNBRW weights computation completed in {time.time() - start_time:.2f} seconds"f"Total normalized weight: {sum(G_copy[u][v][weight_attr] for u, v in G_copy.edges()):.4f}")
    
    return G_copy
