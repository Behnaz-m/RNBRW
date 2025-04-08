# RNBRW
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
@article{moradi2018new,
	title={New methods for incorporating network cyclic structures to improve community detection},
	author={Moradi, Behnaz and Shakeri, Heman and Poggi-Corradini, Pietro and Higgins, Michael},
	journal={arXiv preprint arXiv:1805.07484},
	year={2018}
}
```

## License
MIT
