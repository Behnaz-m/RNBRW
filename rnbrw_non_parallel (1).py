#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
import sys
#np.set_printoptions(threshold=np.nan)
def walk_hole_E(G):
    """
    Random walk that traps at a hole and weights the terminal edge
    """
    m = 0
    for a,b in G.edges():
        G[a][b]['enum'] = m 
        m += 1
    m = G.number_of_edges()  
    E =list(G.edges())
    T = np.zeros(m, dtype= int)    
    S = nx.number_of_edges(G)
#     S = np.floor(m)
    
    #print(np.random.choice(m, 2))
    L = np.random.choice(m, S)
    E_sampled = [E[i] for i in np.ndarray.tolist(L)]
    #E_sampled = np.random.choice(E, S)
    for x,y in E_sampled:
        for u,v in [(x,y), (y,x)]:

            walk = [u,v]
            while True:
                nexts = list(G.neighbors(v))
                try:
                    nexts.remove(u)  #removing the walk head, for the next step
                except ValueError:  #check if head is  in the next step. Avoid problem in the start
                    pass

                if nexts == []:#if no hole found
                    break

                next = np.random.choice(nexts) #next step of the walker
                if next in walk: #if encounters a hole

                    T[  G[v][next]['enum']  ] += 1
                    break

                walk.append(next) #still no hole, prepare for the next step
                u = v
                v = next
    return T

def cycle_prop_E(G, nsim = 2, parallel = True):
    """
    Repeating walk_hole nsim times\n",
    """
    for u,v in G.edges():
        G[u][v]["ret"] = 0.01 #starting small weights on the edges\n",
    for u,v in G.edges():
        G[u][v]["ret_n"] = 0.01 #starting small weights on the edges\n",
        
    m = G.number_of_edges()       
    T = np.zeros(m, dtype= int)
    for i in range(nsim):
        T += walk_hole_E(G)
        
        
    m = 0       
    for a,b in G.edges():
        G[a][b]['ret'] = T[m] 
        m += 1
#     m = 0
#     for a,b in G.edges():
#         G[a][b]['ret_n'] = T[m]/1000.;
#         m += 1 
    #wmax = max([G[u][v]['ret'] for u,v in G.edges()])  #Finding the maximum found edge weight\n",
    #for u,v in G.edges():
        #G[u][v]["ret_n"] = 1.*G[u][v]["ret_n"]/wmax #Normalizing the weights\n",
    return G
G = nx.read_edgelist('bii_data/edgelist_0819.txt', nodetype=str, data=(('weight',float),))
cycle_prop_E(G, nsim = 1, parallel = True)
RNBRW_weights = nx.get_edge_attributes(E,'ret')
SR = sum(RNBRW_weights.values())
for k,v in RNBRW_weights.items():
    RNBRW_weights[k]=  RNBRW_weights[k]/SR
with open('RNBRW_weights.pickle', 'wb') as handle:
    pickle.dump(RNBRW_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    


# In[ ]:




