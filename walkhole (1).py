import networkx as nx
import numpy as np
import sys
#np.set_printoptions(threshold=np.nan)


def walk_hole(G, edges):
    """
    Random walk that traps at a hole and weights the terminal edge
    """
    m = 0
    for a,b in G.edges():
        G[a][b]['enum'] = m 
        m += 1

    m = G.number_of_edges()       
    T = np.zeros(m, dtype= int)    
    S= 6000
    #S = np.floor(m/100.)
    #S = 30
    E = G.edges()
    E_sampled = [E[i] for i in np.random.choice(range(m), S)]
    
    for x,y in E_sampled:
        for u,v in [(x,y), (y,x)]:

            walk = [u,v]
            while True:
                nexts = G.neighbors(v)
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
    G = nx.read_edgelist('/sfs/qumulo/qhome/bm7mp/bii_data/edgelist_0819.txt', nodetype=str, data=(('weight',float),))
    np.savetxt('/sfs/qumulo/qhome/bm7mp/OSS/rnbrw/Github_'+sys.argv[1], walk_hole(G, G.edges()).astype(int), fmt='%i')  

