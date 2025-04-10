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

def assign_rnbrw_weights(G, T, attr="ret"):
    """
    Assign RNBRW edge weights (raw and normalized) from a precomputed vector T.

    Parameters
    ----------
    G : networkx.Graph
        The input graph with edges already enumerated.
    T : array-like
        Vector of edge-wise weights (e.g., sum of walk_hole_E outputs).
    attr : str, optional
        Name of the edge attribute to assign (default is 'ret').

    Returns
    -------
    G : networkx.Graph
        Graph with updated edge attributes: attr and attr + '_n'.
    """
    total = T.sum() or 1
    for i, (u, v) in enumerate(G.edges()):
        G[u][v][attr] = T[i]
        G[u][v][f"{attr}_n"] = T[i] / total
    return G
