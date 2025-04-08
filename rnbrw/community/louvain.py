"""Community detection algorithms using RNBRW weights."""

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
