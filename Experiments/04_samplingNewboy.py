from NetSwitchAlgs import *
import matplotlib.pyplot as plt

target_r = -0.3
sigma = 0.001
net = ig.Graph.Erdos_Renyi(n=50, p=.2)
Snet = NetSwitch(net, pos_only=False)

r_seq = []
for swt_no in range(2000):
    cur_r = Snet.assortativity_coeff()
    r_seq.append(cur_r)
    delta_r = target_r - cur_r
    p = np.exp(-((delta_r/sigma)**2))/2
    switch_dir = bool(np.sign(delta_r)+1)
    #print(target_r, cur_r, p, switch_dir)
    if np.random.rand() < p:
        Snet.switch_A(count=1, pos=(not switch_dir))
        #print('pos', not switch_dir)
    else:
        Snet.switch_A(count=1, pos=switch_dir)
        #print('pos', switch_dir)

plt.figure()
plt.plot(np.arange(len(r_seq)), r_seq)
#plt.imshow(Snet.A)
plt.show()