from NetSwitchAlgs import *
import pickle
import random
import igraph as ig
import os.path
import matplotlib.pyplot as plt
import time
import numpy as np
import networkx as nx

n = 128
p = 0.10
model = 'ER'
# random.seed(0)
# if model == 'ER':
#     ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
#     S = NetSwitch(ERgraph)
# elif model == 'BA':
#     BAgraph = ig.Graph.Barabasi(n=n, m=int(np.round(p * (n - 1) / 2)))
#     S = NetSwitch(BAgraph)
#
# ToMatch = nx.Graph()
# degree_margin = 0
# for i in range(S.n):
#     if not ToMatch.has_node(str(i) + '_0'):
#         ToMatch.add_nodes_from([str(i) + '_' + str(node_copy_idx) for node_copy_idx in range(S.deg[i])])
#     for j in range(i+1, min([S.n, i+S.deg[i]+1+degree_margin])):
#         if not ToMatch.has_node(str(j) + '_0'):
#             ToMatch.add_nodes_from([str(j) + '_' + str(node_copy_idx) for node_copy_idx in range(S.deg[j])])
#         thisPossibleEdge = [str(i) + '__' + str(j), str(j) + '__' + str(i)]
#         ToMatch.add_nodes_from(thisPossibleEdge)
#         ToMatch.add_edge(thisPossibleEdge[0], thisPossibleEdge[1], weight=S.deg[i]*S.deg[j])
#         for node_copy_idx in range(S.deg[i]):
#             ToMatch.add_edge(thisPossibleEdge[0], str(i)+'_'+str(node_copy_idx), weight=S.deg[i]*S.deg[j])
#         for node_copy_idx in range(S.deg[j]):
#             ToMatch.add_edge(thisPossibleEdge[1], str(j)+'_'+str(node_copy_idx), weight=S.deg[i]*S.deg[j])
#
# # ToMatch = ig.Graph()
# # edge_weights = []
# # for i in range(S.n):
# #     if ToMatch.vcount()==0 or str(i) + '_0' not in ToMatch.vs['name']:
# #         ToMatch.add_vertices([str(i) + '_' + str(node_copy_idx) for node_copy_idx in range(S.deg[i])])
# #     for j in range(i+1, min([S.n, i+S.deg[i]+1])):
# #         if str(j)+'_0' not in ToMatch.vs['name']:
# #             ToMatch.add_vertices([str(j) + '_' + str(node_copy_idx) for node_copy_idx in range(S.deg[j])])
# #         thisPossibleEdge = [str(i) + '__' + str(j), str(j) + '__' + str(i)]
# #         ToMatch.add_vertices(thisPossibleEdge)
# #         ToMatch.add_edge(thisPossibleEdge[0], thisPossibleEdge[1])
# #         edge_weights.append(0)
# #         for node_copy_idx in range(S.deg[i]):
# #             ToMatch.add_edge(thisPossibleEdge[0], str(i)+'_'+str(node_copy_idx))
# #             edge_weights.append(S.deg[i]*S.deg[j])
# #         for node_copy_idx in range(S.deg[j]):
# #             ToMatch.add_edge(thisPossibleEdge[1], str(j)+'_'+str(node_copy_idx))
# #             edge_weights.append(S.deg[i]*S.deg[j])
# # ToMatch.es['weight'] = edge_weights
# # G = ToMatch.to_networkx()
#
# matching = nx.max_weight_matching(ToMatch, maxcardinality=False)
# matchingDict = {}
# for matched_edge in matching:
#     e, v = -1, -1
#     for node_string in matched_edge:
#         node_split = node_string.split("_")
#         if len(node_split)==3:
#             int_nodes = [int(node_split[0]), int(node_split[2])]
#             e = (min(int_nodes), max(int_nodes))
#         if len(node_split)==2:
#             v = int(node_split[0])
#     if v != -1:
#         if e not in matchingDict:
#             matchingDict[e]=0
#         matchingDict[e] += 1
#     else:
#         1
#
# optimalGraph = nx.Graph()
# optimalGraph.add_nodes_from(range(S.n))
# for e in matchingDict:
#     if matchingDict[e] == 2:
#         optimalGraph.add_edge(e[0], e[1])
#
# #print(sorted([d for v, d in optimalGraph.degree()], reverse=True))
# #print(list(S.deg))
# optG_degseq = np.array(sorted([d for v, d in optimalGraph.degree()], reverse=True))
# #print (S.deg.size, optG_degseq.size)
# print(np.sum(np.abs(optG_degseq - S.deg)))
#
# #sorted_nodes = sorted(optimalGraph, key=lambda x: (optimalGraph.degree(x), np.sum([optimalGraph.degree(i) for i in optimalGraph.neighbors(x)])), reverse=True)
# #print([optimalGraph.degree(i) for i in optimalGraph.neighbors(0)])
# #adj_matrix = nx.adjacency_matrix(optimalGraph, nodelist=sorted_nodes).toarray()
# adj_matrix = nx.adjacency_matrix(optimalGraph).toarray()

res_fname = "NetPosSwitchingResults/MAX_"+model+"_N"+str(int(n))+"_p"+str(int(p*100))+".pkl"
# with open(res_fname, 'wb') as out_f:
#     pickle.dump(adj_matrix, out_f)
with open(res_fname, 'rb') as in_f:
    adj_matrix = pickle.load(in_f)
plt.imshow(adj_matrix)
plt.show()


