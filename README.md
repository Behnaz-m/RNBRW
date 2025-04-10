# RNBRW

[![PyPI version](https://badge.fury.io/py/rnbrw.svg)](https://pypi.org/project/rnbrw/)

**RNBRW** (Renewal Non-Backtracking Random Walks) is a Python package for estimating edge-level importance in networks using random walks that restart upon cycle closure. These weights can be used to improve community detection algorithms like Louvain.

Based on:

> **Moradi, B.**, **Shakeri, H.**, **Poggi-Corradini, P.**, & **Higgins, M.**  
> *A new method for incorporating network cyclic structures to improve community detection*  
> [arXiv:1805.07484](https://arxiv.org/abs/1805.07484)

---

##  Installation

```bash
pip install rnbrw
```

## Features
- Parallel RNBRW edge weight estimation
- Seamless integration with Louvain
- Based on [Moradi-Jamei et al., 2019](https://arxiv.org/abs/1805.07484)

## Parallelization & HPC Useage

rnbrw supports parallel execution of the RNBRW simulations using joblib. This allows for efficient simulation on multi-core machines or High-Performance Computing (HPC) clusters.

You can control parallel execution using the n_jobs parameter:

- Local machine:
Set n_jobs=-1 to use all available CPU cores, or specify the exact number of cores to use (e.g., n_jobs=4).

- High-Performance Computing (HPC):
When running on an HPC cluster, n_jobs can be tuned according to the allocated CPUs in your job script. For best performance, align n_jobs with the number of cores requested via sbatch, qsub, or your cluster’s job scheduler.

### HPC Usage


Step 1: Run RNBRW Walks in Parallel Jobs

SLURM Example
Here’s a basic SLURM job array script:rnbrw_job.sh
```bash
#!/bin/bash
#SBATCH --job-name=rnbrw_walk
#SBATCH --output=logs/rnbrw_%A_%a.out
#SBATCH --error=logs/rnbrw_%A_%a.err
#SBATCH --array=0-19              # 20 total jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

module load python/3.10
source activate rnbrw-env

# Run the Python script with job array index
python run_rnbrw_batch.py $SLURM_ARRAY_TASK_ID


```


```python
import sys
import numpy as np
import networkx as nx
from rnbrw.weights import compute_weights

job_id = int(sys.argv[1])
seed = 1000 + job_id

# Load the graph (shared across all jobs)
G = nx.read_gpickle("mygraph.gpickle")

# Single walk using only_walk mode
G = compute_weights(G, nsim=1, seed=seed, only_walk=True)

# Extract edge counts
m = G.number_of_edges()
T = np.zeros(m)
for u, v in G.edges():
    T[G[u][v]['enum']] = G[u][v]['ret']

np.save(f"T_partial_{job_id}.npy", T)

```


Step 2 – Aggregate outputs (on head node):

```python
import numpy as np

T_total = sum(np.load(f"T_partial_{i}.npy") for i in range(num_jobs))

```

Step 3 –  Assign Weights to Graph

```python
import networkx as nx
import numpy as np
from rnbrw.utils import assign_rnbrw_weights

G = nx.read_gpickle("mygraph.gpickle")
T_total = np.load("T_total.npy")

# Assign raw + normalized weights to the graph
G = assign_rnbrw_weights(G, T_total)

```

Step 4: Run Louvain

```python
from rnbrw.community import detect_communities_louvain

partition = detect_communities_louvain(G, weight_attr='ret_n')

```

This makes rnbrw especially suitable for research environments where cycles and edge roles must be computed across very large networks.


## Local Usage
Use compute_weights directly with multi-threading:
```python
import networkx as nx
from rnbrw.weights import compute_weights
from rnbrw.community import detect_communities_louvain

# Create or load a graph
G = nx.karate_club_graph()

# Compute RNBRW weights
# Recommendation: nsim should be at least the number of edges in G
G = compute_weights(G, nsim=G.number_of_edges(), n_jobs=4)

# Edge weights (normalized)
weights = [G[u][v]['ret_n'] for u, v in G.edges()]

# Detect communities
from rnbrw.community import detect_communities_louvain

partition = detect_communities_louvain(G, weight_attr='ret_n')

```
## API Reference

```python
G_weighted = compute_weights(
    G,               # networkx.Graph
    nsim=None,       # Optional[int], defaults to factor * m (num edges)
    factor=1.0,      # float, multiplies number of edges to compute default nsim
    seed=None,       # Optional[int], random seed for reproducibility
    n_jobs=1,        # int, number of parallel jobs (-1 = all CPUs)
    init_weight=0.001,# float, initial placeholder edge weights
    only_walk=False  # bool, run single walk (no aggregation) for HPC
)

```
 Simulates RNBRW on graph G to assign edge importance scores as weights.

## Parameters for `compute_weights`

| Parameter     | Type              | Default      | Description                                                                 |
|---------------|-------------------|--------------|-----------------------------------------------------------------------------|
| `G`           | `networkx.Graph`  | *required*   | Input undirected graph.                                                     |
| `nsim`        | `int or None`     | `None`       | Number of RNBRW simulations. If `None`, it defaults to `factor × m`, where `m` is the number of edges. |
| `factor`      | `float`           | `1.0`        | Scaling factor to set `nsim` dynamically based on graph size (`m`).         |
| `n_jobs`      | `int`             | `1`          | Number of parallel jobs (-1 uses all available CPUs).                       |
| `seed`        | `int or None`     | `None`       | Random seed for reproducibility. Each simulation is seeded with `seed + i`. |
| `init_weight` | `float`           | `0.01`       | Initial placeholder weight for each edge before running RNBRW.              |
| `only_walk`   | `bool`            | `False`      | If `True`, performs a single walk without aggregating weights (for HPC use).|

### Notes
- **Recommended**: For stable and convergent RNBRW edge weights, set `nsim` approximately equal to `m`, the number of edges in the graph (e.g., `nsim ≈ m`).
- If `only_walk=True`, the function will return the walk output without updating graph weights — useful for splitting across HPC batch jobs manually.


```python
detect_communities_louvain(G, weight_attr='ret_n')
```
Runs Louvain on G using edge weights.

| Parameter     | Type              | Description                                                     |
|---------------|-------------------|-----------------------------------------------------------------|
| `G`           | `networkx.Graph`  | Weighted input graph                                            |
| `weight_attr` | `str`             | Edge weight attribute used for Louvain (default = `'ret_n'`)   |
 |
```python
normalize_edge_weights(G, weight='ret')
```
Normalizes the weights to sum to 1 across all edges.

| Parameter     | Type              | Description                                                     |
|---------------|-------------------|-----------------------------------------------------------------|
| `G`           | `networkx.Graph`  | Graph whose edge weights are to be normalized                  |
| `weight`      | `str`             | Edge attribute to normalize (default = `'ret'`)                |



If you are running RNBRW simulations in parallel (e.g., on an HPC cluster), you can aggregate the walk counts and assign RNBRW weights manually using the function below.

```python
# After accumulating total walk counts across jobs:
G_weighted = assign_rnbrw_weights(G, T_total)
# G must have 'enum' attributes on edges
```

| Parameter      | Type              | Description                                                                 |
|----------------|-------------------|-----------------------------------------------------------------------------|
| `G`            | `networkx.Graph`  | Input graph with edges having `'enum'` attribute for indexing               |
| `T_total`      | `np.ndarray`      | Array of edge hit counts from RNBRW simulations (same order as `'enum'`)    |

> Use this when you aggregate RNBRW cycle counts manually and want to assign weights post-hoc.


 ## Citation
If you use this package in your research, please cite:


@article{moradi2018new,
  title={A new method for incorporating network cyclic structures to improve community detection},
  author={Moradi, Behnaz and Shakeri, Heman and Poggi-Corradini, Pietro and Higgins, Michael},
  journal={arXiv preprint arXiv:1805.07484},
  year={2018}
}
Or use the “Cite this repository” button above.

## License
This project is licensed under the MIT License © 2025 Behnaz Moradi-Jamei.
## Documentation
Full documentation is available at [Read the Docs](https://rnbrw.readthedocs.io).


