from NetSwitchAlgs import *
import pickle
import random
import igraph as ig
import matplotlib.pyplot as plt


# n = 100
# p = 0.2
# random.seed(0)
# ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
# S = NetSwitch(ERgraph)
# result = {0.4: [], 0.7: [], 0.95: []}
# for pos_p in result:
#     for s_no in range(50000):
#         S.XBS(pos_p=pos_p, count=1)
#         result[pos_p].append((S.swt_done, S.assortativity_coeff(), S.total_checkers()))
#         if s_no > 0 and s_no % 10000 == 0:
#           print(s_no, 'switches with p =', pos_p)
# with open('XBS_ER_N100_p20.pkl', 'wb') as out_f:
#     pickle.dump(result, out_f)

with open('XBS_ER_N100_p20.pkl', 'rb') as in_f:
    result = pickle.load(in_f)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot([i[1] for i in result[0.4]], c='#EDC951', label=r'$p_{XBS}=0.4$')
plt.plot([i[1] for i in result[0.7]], c='#FD8060', label=r'$p_{XBS}=0.7$')
plt.plot([i[1] for i in result[0.95]], c='#E84258', label=r'$p_{XBS}=0.95$')
plt.ylabel(r'Assortativty $r$')
plt.xlabel(r'Number of switches')
plt.legend(frameon=False, fontsize=9, ncol=2)
plt.xlim([-1000, 20000])
plotLabelY = 0.95*(plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0])+plt.gca().get_ylim()[0]
plt.text(-6000, plotLabelY, 'a', weight='bold', fontsize=24)

plt.subplot(1, 2, 2)
plt.plot([i[2]/10000 for i in result[0.95]], c='#E84258')
plt.plot([i[2]/10000 for i in result[0.7]], c='#FD8060')
plt.plot([i[2]/10000 for i in result[0.4]], c='#EDC951')
plt.ylabel(r'Remaining switches')
plt.xlabel(r'Number of switches')
plt.text(-1000, 30.2, r'$\times 10^4$')
plotLabelY = 0.95*(plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0])+plt.gca().get_ylim()[0]
plt.text(-6000, plotLabelY, 'b', weight='bold', fontsize=24)
plt.xlim([-1000, 20000])
plt.tight_layout()
plt.show()
