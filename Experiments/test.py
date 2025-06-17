import numpy as np
from NetSwitchAlgs import *
import matplotlib.pyplot as plt
import pickle
import time


start = time.time()
sample_count = 2000
target_r = [.4, .6]
''' Load the network and make SwitchNet Obj'''

# filename = 'configs_custom_n9_m9_noDB.pkl'
# with open(filename, 'rb') as in_f:
#     data = pickle.load(in_f)
np.random.seed(0)
net = ig.Graph.Erdos_Renyi(n=64, p=.1)
Snet = NetSwitch(net, pos_only=False)

''' Sampling Procedure'''
cur_r = Snet.assortativity_coeff()
while not target_r[0] <= cur_r <= target_r[1]:
    print(Snet.total_checkers(pos=True), cur_r)
    swt = Snet.find_random_checker(pos=True if cur_r < target_r[0] else False)
    Snet.switch(swt)
    cur_r = Snet.assortativity_coeff()

x_pos, x_neg = Snet.total_checkers(pos=True), Snet.total_checkers(pos=False)
cur_s = x_pos + x_neg
posSwt_ratio = x_pos / cur_s
samples = [cur_r]
iterNo = 0
while iterNo < sample_count:
    swt = Snet.find_random_checker(pos=True if np.random.rand()<posSwt_ratio else False)
    Snet.switch(swt)
    nxt_r = Snet.assortativity_coeff()
    nxt_s = Snet.total_checkers(both=True)

    if not target_r[0] <= nxt_r <= target_r[1]:
        accept_pr = 0
    else:
        accept_pr = min([1, cur_s / nxt_s])

    if np.random.rand() >= accept_pr:
        Snet.switch(swt)
    else:
        cur_r = nxt_r
        cur_s = nxt_s
        posSwt_ratio = Snet.total_checkers(pos=True) / cur_s
        samples.append(cur_r)
        iterNo += 1
        if iterNo % 1000 == 0:
            print(iterNo)

plt.figure()
plt.hist(samples, bins=100)
plt.show()
end = time.time()
print(end-start)