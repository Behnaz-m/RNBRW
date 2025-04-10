"""Implementation of Renewal Non-Backtracking Random Walk (RNBRW) weight computation."""

import numpy as np
import networkx as nx
import time
from joblib import Parallel, delayed
#from .walk import walk_hole_E  # or wherever you define walk_hole_E

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
    #sample with replacement
    S=m
    L = np.random.choice(m, S, replace=True)
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
def compute_weights(
    G, nsim=None, seed=None, factor=1.0,  n_jobs=1, init_weight=0.01, only_walk=False
):
    """
    Compute RNBRW edge weights using either full cycle propagation or walk_hole_E only.

    Parameters
    ----------
    G : networkx.Graph
        The graph on which RNBRW is run
    nsim : int
        Number of RNBRW simulations
    seed : int or None
        Random seed for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 uses all CPUs)
    init_weight : float
        Initial placeholder weight for all edges before computation
    only_walk : bool
        If True, run only the walk_hole_E logic once (for HPC users managing batching externally)

    Returns
    -------
    G : networkx.Graph
        Graph with updated RNBRW weights in 'ret' and 'ret_n'
        
References
    ----------
    .. [1] Moradi, Behnaz and Shakeri, Heman and Poggi-Corradini, Pietro and Higgins, Michael(2019).
           A new method for incorporating network cyclic structures to improve community detection.
           arXiv preprint arXiv:1805.07484.   
    

    """
    m = G.number_of_edges()
    if nsim is None:
        nsim = int(m*factor)  # Default to number of edges for convergence
        
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['enum'] = i
        G[u][v]['ret'] = init_weight
        G[u][v]['ret_n'] = init_weight

    if only_walk:
        # No accumulation, just one walk
        T = walk_hole_E(G, seed=seed)
    else:
        seeds = [(seed + i) if seed is not None else None for i in range(nsim)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(walk_hole_E)(G, s) for s in seeds
        )
        T = sum(results)

    total = T.sum() if T.sum() > 0 else 1

    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['ret'] = T[i]
        G[u][v]['ret_n'] = T[i] / total

    return G

def assign_rnbrw_weights(G, T, weight_attr='ret_n'):
    """
    Assign normalized RNBRW weights to graph edges from a hit count array.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose edges have been enumerated (i.e., G[u][v]['enum'] exists).
    T : np.ndarray
        Array of cycle counts for each edge (same length and order as G.edges()).
    weight_attr : str
        Attribute name for normalized weights (default='ret_n').

    Returns
    -------
    networkx.Graph
        Graph with assigned edge weights in attribute `weight_attr`.
    """
    total = T.sum() if T.sum() > 0 else 1  # avoid division by zero
    for u, v, data in G.edges(data=True):
        idx = data.get("enum", None)
        if idx is not None and idx < len(T):
            data["ret"] = T[idx]
            data[weight_attr] = T[idx] / total
    return G


