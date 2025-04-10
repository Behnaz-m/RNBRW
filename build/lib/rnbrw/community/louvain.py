"""Community detection using RNBRW edge weights."""

import community as community_louvain
import networkx as nx

def detect_communities_louvain(G, weight_attr='ret_n', random_state=None):
    """
    Apply Louvain community detection using RNBRW edge weights.

    Parameters
    ----------
    G : networkx.Graph
        Graph with RNBRW weights (e.g., from compute_weights or HPC aggregation)
    weight_attr : str, optional
        Name of the edge attribute containing weights (default = 'ret_n')
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary mapping node → community ID

    Raises
    ------
    ValueError
        If the specified edge weight attribute is missing

    Notes
    -----
    Requires the `python-louvain` package.
    """
    # Sanity check: ensure weights exist
    for u, v in G.edges():
        if weight_attr not in G[u][v]:
            raise ValueError(f"Edge ({u}, {v}) missing attribute '{weight_attr}'. "
                             "Ensure RNBRW weights are computed first.")

    return community_louvain.best_partition(G, weight=weight_attr, random_state=random_state)
