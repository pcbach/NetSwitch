from NetSwitchAlgs import *
import pickle
import random
import igraph as ig
import os.path
import matplotlib.pyplot as plt

n = 25
p = .2
swt_batch_count = 10

random.seed(0)
ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
S = NetSwitch(ERgraph)
rOrdr = [(S.swt_done, S.assortativity_coeff())]
while True:
    ret = S.switch_A(alg='ORDR', count=swt_batch_count)
    rOrdr.append((S.swt_done, S.assortativity_coeff()))
    if ret != -1:
        break

S.checkercount_matrix()
print(S.total_checkers())
S.checkercount_matrix(count_upper=False)
print(S.total_checkers())

random.seed(0)
ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
S = NetSwitch(ERgraph)
rXbs = [(S.swt_done, S.assortativity_coeff())]
while True:
    ret = S.XBS(pos_p=1.0, count=swt_batch_count)
    rXbs.append((S.swt_done, S.assortativity_coeff()))
    if ret != -1:
        break

S.checkercount_matrix()
print(S.total_checkers())
S.checkercount_matrix(count_upper=False)
print(S.total_checkers())

plt.imshow(S.A)
plt.show()

