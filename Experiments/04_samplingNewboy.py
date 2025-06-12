from NetSwitchAlgs import *
import matplotlib.pyplot as plt
import pickle


def normal(r, mu, sigma):
    # return 1
    #return np.exp(((r-mu)/sigma)**2/(-2))/(sigma*np.sqrt(2*np.pi))
    return 1 if mu-sigma<r<mu+sigma else 0

target_r = 0.4
sigma = 0.05
net = ig.Graph.Erdos_Renyi(n=32, p=.15)
Snet = NetSwitch(net, pos_only=False)
with open('testpickleHighR.pkl', 'rb') as in_f:
    Snet = pickle.load(in_f)

cur_r = Snet.assortativity_coeff()
cur_s = Snet.total_checkers(both=True)
posSwt_ratio = Snet.total_checkers(pos=True)/cur_s
x_pos, x_neg = Snet.total_checkers(pos=True), Snet.total_checkers(pos=False)
samples = [cur_r]
for iter in range(20000):
    swt = Snet.find_random_checker(pos=True if np.random.rand()<posSwt_ratio else False)
    Snet.switch(swt)
    nxt_r = Snet.assortativity_coeff()
    nxt_s = Snet.total_checkers(both=True)
    magic_ratio = 1
    #print(magic_ratio)
    if normal(nxt_r, target_r, sigma) == 0 and normal(cur_r, target_r, sigma) == 0:
        apr = 1
    elif normal(cur_r, target_r, sigma)==0:
        apr = 100000
    else:
        apr = (normal(nxt_r, target_r, sigma)/normal(cur_r, target_r, sigma))
    #(normal(nxt_r, target_r, sigma)/normal(cur_r, target_r, sigma))
    #print(iter, apr, nxt_r)
    # if apr == 0:
    #     1/0
    accept_pr = min([1, apr * (cur_s/nxt_s)])
    #print('accept pr', accept_pr)
    if np.random.rand() >= accept_pr:
        Snet.switch(swt)
    else:
        cur_r = nxt_r
        cur_s = nxt_s
        posSwt_ratio = Snet.total_checkers(pos=True) / cur_s
        x_pos, x_neg = Snet.total_checkers(pos=True), Snet.total_checkers(pos=False)
        samples.append(cur_r)
        #print(cur_r)
# with open('testpickleHighR.pkl', 'wb') as out_f:
#     pickle.dump(Snet, out_f)
plt.figure(figsize=(8, 3.5))
plt.subplot(1, 2, 1)
plt.plot(samples)
plt.plot([-1, len(samples)+1], [target_r, target_r])
plt.xlabel('Sample No.')
plt.ylabel('Degree Assortativity')
plt.subplot(1, 2, 2)
plt.hist(samples, bins=100, density=True)
#plt.plot(np.linspace(0, .4, 100), normal(np.linspace(0, .4, 100), target_r, sigma))
plt.xlabel('Degree Assortativity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
1/0
swt = self.find_random_checker(pos=pos)

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