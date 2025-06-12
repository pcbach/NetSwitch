import numpy as np
import igraph as ig
import pickle
import matplotlib.pyplot as plt


def rec_graph_gen(deg_seq, edges):
    stubby_nodes = np.nonzero(deg_seq)[0]
    if stubby_nodes.size == 0:
        print(edges)
        return edges
    else:
        i = int(stubby_nodes[0])
        configs = []
        if  stubby_nodes[1:].size > 0:
            for j in stubby_nodes[1:]:
                #print(i,j, deg_seq, edges)
                if (i,j) in edges:
                    continue
                deg_seq_new = np.copy(deg_seq)
                deg_seq_new[i] -= 1
                deg_seq_new[j] -= 1
                if np.sum(deg_seq_new) >= 0:
                    #print(deg_seq_new, i, j)
                    ret = rec_graph_gen(deg_seq_new, edges+[(i,j)])
                    if ret is not None:
                        configs.append(ret)
                elif np.sum(deg_seq_new) == 0:
                    1#print(deg_seq_new)
                    #print(edges)
        print(configs)
        return None

def r_coeff(n, es):
    g = ig.Graph(n)
    g.add_edges(list(es))
    return g.assortativity_degree()

def rec_graph_gen1(deg_seq, edges):
    stubby_nodes = np.nonzero(deg_seq)[0]
    i = int(stubby_nodes[0])
    configs = []
    if  stubby_nodes[1:].size > 0:
        for j in stubby_nodes[1:]:
            #print(i,j, deg_seq, edges)
            if (i,j) in edges:
                continue
            deg_seq_new = np.copy(deg_seq)
            deg_seq_new[i] -= 1
            deg_seq_new[j] -= 1
            if np.sum(deg_seq_new) > 0:
                #print(deg_seq_new, i, j)
                ret = rec_graph_gen1(deg_seq_new, edges+[(i,j)])
                if ret is not None:
                    for elist in ret:
                        if set(elist) not in configs:
                            #print(set(elist))
                            configs.append(set(elist))
            elif np.sum(deg_seq_new) == 0:
                new_elist = set(edges+[(i,j)])
                if new_elist not in configs:
                    #print(new_elist)
                    configs.append(new_elist)
                    #print(r_coeff(len(deg_seq), new_elist))
    return configs

n1, n2 = 6, 6
d1, d2 = 2, 3
degs = np.concatenate([np.repeat(d1, n1), np.repeat(d2, n2)])
if ig.is_graphical(degs):
    configs = rec_graph_gen1(degs, [])

    g = ig.Graph(n=len(degs), directed=False)
    rList = []
    for config in configs:
        g.add_edges(list(config))
        rList.append(g.assortativity_degree())
        g.delete_edges(list(config))
    with open('configs_n6_23.pkl', 'wb') as out_f:
        pickle.dump({'e':configs, 'r':rList}, out_f)
    plt.figure(figsize=(5,3))
    plt.hist(rList, bins=100)
    plt.xlabel('Assortativity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()