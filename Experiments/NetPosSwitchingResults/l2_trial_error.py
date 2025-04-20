from NetSwitchAlgs import *
import pickle
import random
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import copy

n = 128
p = 0.05
random.seed(0)
ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
S = NetSwitch(ERgraph)
Aorg = copy.copy(S.A)
data = [(S.swt_done, S.lev(), S.l2(normed=True))]
while True:
    swt_num = S.switch_A(alg='SWPC', count=10)
    data.append((S.swt_done, S.lev(), S.l2(normed=True)))
    if S.swt_done % 100 == 0:
        print(S.swt_done)
    if swt_num != -1:
        break

cmap = colors.ListedColormap(['white', 'tab:blue'])
plt.figure(figsize=(15,3))
plt.subplot(1, 3, 1)
plt.imshow(Aorg, cmap=cmap)
plt.subplot(1, 3, 2)
plt.plot([i[0] for i in data], [100*(i[1]/data[0][1]-1) for i in data])
plt.plot([i[0] for i in data], [100*(i[2]/data[0][2]-1) for i in data])
plt.subplot(1, 3, 3)
plt.imshow(S.A, cmap=cmap)
plt.tight_layout()
plt.show()
#result[pos_p].append((S.swt_done, S.assortativity_coeff(), S.total_checkers()))
#if s_no > 0 and s_no % 10000 == 0:
#  print(s_no, 'switches with p =', pos_p)
