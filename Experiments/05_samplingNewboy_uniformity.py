import numpy as np
from NetSwitchAlgs import *
import matplotlib.pyplot as plt
import pickle


sample_count = 20480000

''' Load the network and make SwitchNet Obj'''

filename = 'configs_custom_n9_m9_noDB.pkl'
with open(filename, 'rb') as in_f:
    data = pickle.load(in_f)

if sample_count not in data['s']:

    if not ig.is_graphical(data['d']):
        raise 'The input degree sequence is NOT GRAPHIC!'
    net = ig.Graph.Degree_Sequence(data['d'], method='vl')
    Snet = NetSwitch(net, pos_only=False)

    all_configs = data['e']
    config_index_map = {frozenset(s): i for i, s in enumerate(all_configs)}
    # all_configs = []
    # for eList in all_elists:
    #     all_configs.append(sorted(eList, key=lambda x:(x[0], x[1])))
    # all_configs = np.array(all_configs)
    # del eList


    ''' Sampling Procedure'''
    cur_r = Snet.assortativity_coeff()
    cur_s = Snet.total_checkers(both=True)
    posSwt_ratio = Snet.total_checkers(pos=True)/cur_s
    x_pos, x_neg = Snet.total_checkers(pos=True), Snet.total_checkers(pos=False)

    samples = [cur_r]
    config_sample_freq = [0 for _ in range(len(all_configs))]
    config_sample_freq[config_index_map.get(frozenset(Snet.get_edges()), -1)] += 1

    iterNo = 0
    while iterNo < sample_count:
        swt = Snet.find_random_checker(pos=True if np.random.rand()<posSwt_ratio else False)
        Snet.switch(swt)
        nxt_r = Snet.assortativity_coeff()
        nxt_s = Snet.total_checkers(both=True)

        accept_pr = 1#min([1, cur_s/nxt_s])

        if np.random.rand() >= accept_pr:
            Snet.switch(swt)
        else:
            cur_r = nxt_r
            cur_s = nxt_s
            posSwt_ratio = Snet.total_checkers(pos=True) / cur_s
            x_pos, x_neg = Snet.total_checkers(pos=True), Snet.total_checkers(pos=False)
            samples.append(cur_r)
            config_sample_freq[config_index_map.get(frozenset(Snet.get_edges()), -1)] += 1
            iterNo += 1

    with open(filename, 'wb') as out_f:
        data['s'][sample_count] = {'r': samples, 'e': config_sample_freq}
        pickle.dump(data, out_f)

all_configs_r = data['c']
sorted_config_idxs = np.argsort(all_configs_r)
#print(np.array(data['c'])[sorted_config_idxs])


plt.figure(figsize=(12, 3.5))
plt.subplot(1, 3, 1)
#plt.hist(data['r'], bins=20, density=False)
_ = plt.hist([data['s'][10240000]['r'], data['r']] , bins=20, density=True, color=['#669bbc', '#c1121f'], label=['Actual', 'Sampled'])
plt.xlim([-1, +1])
#plt.plot(np.linspace(0, .4, 100), normal(np.linspace(0, .4, 100), target_r, sigma))
plt.xlabel('Degree Assortativity')
plt.ylabel('Probability density')
plt.legend(frameon=False)
plt.subplot(1, 3, 2)
colors=['#4cc9f0', '#e9c46a', '#2a9d8f', '#e76f51']
plt.plot(np.array(data['s'][320000]['e'])[sorted_config_idxs]/320000, c=colors[2], marker='o', markersize=3, lw=0, alpha=0.7, label=r'$8\times 10^4$ samples')
plt.plot(np.array(data['s'][1280000]['e'])[sorted_config_idxs]/1280000, c=colors[1], marker='o', markersize=3, lw=0, alpha=0.7, label=r'$32\times 10^4$ samples')
plt.plot(np.array(data['s'][10240000]['e'])[sorted_config_idxs]/10240000, c=colors[3], marker='o', markersize=3, lw=0, alpha=0.7, label=r'$256\times 10^4$ samples')
plt.plot([0, len(data['e'])], [1.0/len(data['e']), 1.0/len(data['e'])], c='k', lw=2, ls='--', label=r'$1/|\mathcal{G}|$')
plt.xlabel('Graph configuration')
plt.ylabel('Sampling probability')
plt.legend(frameon=False)
plt.ylim([0,0.0014])
plt.subplot(1, 3, 3)
plt.plot([2**i*10000 for i in range(11)], [np.var(np.array(data['s'][2**i*10000]['e'])/2**i*10000) for i in range(11)], c='#0a9396', lw=3)
plt.xlabel('Sample count')
plt.ylabel('Frequency variance')
plt.xscale('log')
plt.tight_layout()
plt.show()
