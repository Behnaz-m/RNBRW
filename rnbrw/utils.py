"""Utility functions for RNBRW package."""

def normalize_edge_weights(G, weight='ret'):
    """
    Normalize edge weights in a NetworkX graph so they sum to 1.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    weight : str
        Name of the edge attribute storing the weights (default is 'ret').

    Returns
    -------
    G : networkx.Graph
        The graph with normalized weights.
    """
    total = sum(G[u][v].get(weight, 0.0) for u, v in G.edges())
    if total == 0:
        return G
    for u, v in G.edges():
        G[u][v][weight] = G[u][v].get(weight, 0.0) / total
    return G
def normalize_edge_weights(G, weight='ret'):
    total = sum(G[u][v][weight] for u, v in G.edges())
    if total == 0:
        return G
    for u, v in G.edges():
        G[u][v][weight] /= total
    return G

