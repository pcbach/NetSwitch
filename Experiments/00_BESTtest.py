from NetSwitchAlgs import *
import pickle
import random
import igraph as ig
import os.path
import matplotlib.pyplot as plt
import time

n = 10
p = 0.1
swt_batch_count = 10

# res_fname = "NetPosSwitchingResults/POS_ER_N"+str(int(n))+"_p"+str(int(p*100))+"_bch"+str(swt_batch_count)+".pkl"
#
# with open(res_fname, 'rb') as in_f:
#     result = pickle.load(in_f)

random.seed(0)
ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
S = NetSwitch(ERgraph)
rBest = [(S.swt_done, S.assortativity_coeff())]
st = time.time()
while True:
    ret = S.switch_A(alg='BEST', count=swt_batch_count)
    rBest.append((S.swt_done, S.assortativity_coeff()))
    if ret != -1:
        break
print(time.time()-st)

# plt.imshow(S.A)
# plt.show()


