# RNBRW
 RNBRW quantifies edge importance as the likelihood of an edge completing a cycle in a non-backtracking random walk, providing
a scalable alternative to analyse real-world networks. We describe how RNBRW weights can be coupled with other popular scalable community detection algorithms, such as CNM and Louvain, to improve their performance. Weighting the graph with RNBRW can substantially improve the detection of ground-truth communities.  RNBRW preprocessing step can overcome the problem of detecting small communities known as resolution limit in modularity maximization methods.
 
Each run of the algorithm runs a NBRW and returns the retracing edge (if any) as the sample of retracing edge. The number of times each edge has been retraced by a NBRW is approximately proportional to its retracing probability. The algorithm only requires knowledge the graph
and each run is independent of the other. Therefore, we can collect samples in parallel leading to fast convergence. Each run of the algorithm is comprised of the following steps:
1-Choose a random edge −−→v0v1 in E.
2-Form the walk w = (−−→v0v1).
3- (For k = {1, · · · }) The walker continues her walk from vk to a neighboring node
vk+1 6= vk−1.
4- If v_k+1 has degree 1, return immediately to Step 1. If vk+1 is already in w, return
−−−−→ v_k v_k+1 as the retracing edge and return to Step 1. Otherwise add vk+1 to w and go
to Step 3 incrementing k = k + 1.

Note that these walkers can be initialized independently from each other and retraced edges can be collated at the very end of the process. Therefore,
one can execute this process as an array of jobs on a cluster of computers efficiently.

1- first execute an array of jobs (#m times) for walkhole.py: 
sbatch --array=1-m myjob.sh
2- then collect the jobs using array_wrangle.py and store the final weights:
sbatch Array_wrangle.sh
