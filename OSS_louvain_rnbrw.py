import networkx as nx
import community as community_louvain
import pickle
import pylab as plt
#%matplotlib inline
import matplotlib.pyplot as plt
import random
import numpy as np
from time import clock
from math import *
import wave
from time import clock
import pandas as pd
import seaborn as sns
import igraph as ig
from igraph import *
G = nx.read_edgelist('/sfs/qumulo/qhome/bm7mp/bii_data/edgelist_0819.txt', nodetype=str, data=(('weight',float),))
#gs = nx.connected_components(G)
#node_components = list(gs)
#from networkx.drawing.nx_agraph import graphviz_layout
#import louvain
#g0 = nx.subgraph(G, node_components[0])
X = np.loadtxt('/sfs/qumulo/qhome/bm7mp/OSS/rnbrw/git_total1' , dtype = int ) 
m =  G.number_of_edges()
W = np.zeros(m , dtype = int )
m = 0
for a,b in G.edges(): # numbering the edges
    W[m] = G[a][b]['weight']
    m += 1
W_ret = np.multiply(X, W) # multiply rnbrw weights and repo weights two vector

m = 0
for a,b in G.edges():
    G[a][b]['w_ret'] = W_ret[m]
    m += 1
    
partition = community_louvain.best_partition(G, weight='weight')
def dictinvert(d): #a dict with memebership and nodes of each membership
    inv = {}
    for k, v in d.items():
        keys = inv.setdefault(v, [])
        keys.append(k)
    return inv
membership = dictinvert(partition)
with open('/sfs/qumulo/qhome/bm7mp/OSS/rnbrw/mem_luv_rnbrw.pickle', 'wb') as handle:
    pickle.dump(membership, handle, protocol=pickle.HIGHEST_PROTOCOL)
