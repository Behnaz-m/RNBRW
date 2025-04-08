"""Example usage of the RNBRW package."""

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
