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
    #S = 30
    S = 32000
    #S = nx.number_of_edges(G)
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

if __name__ == "__main__":
    #G =  nx.karate_club_graph()
    G = nx.read_edgelist('/Path/full_edgelist_0819.txt', nodetype=str, data=(('weight',float),))
    #np.savetxt('/home/bm7mp/OS/rnbrw/k_'+sys.argv[1], walk_hole_E(G).astype(int), fmt='%i')  
    np.savetxt('/Path/git_'+sys.argv[1], walk_hole_E(G).astype(int), fmt='%i')  
