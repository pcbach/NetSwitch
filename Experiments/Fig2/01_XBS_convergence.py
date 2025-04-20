from NetSwitchAlgs import *
import pickle
import random
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

# n = 100
# p = 0.2
# random.seed(0)
# ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
# S = NetSwitch(ERgraph)
# result = {0.4: [], 0.7: [], 0.95: []}
# with open('XBS_ER_N100_p20_initial.pkl', 'wb') as out_f:
#     pickle.dump(S.A, out_f)
# for pos_p in result:
#     S = NetSwitch(ERgraph)
#     for s_no in range(50000):
#         S.XBS(pos_p=pos_p, count=1, force_update_N=True)
#         result[pos_p].append((S.swt_done, S.assortativity_coeff(), S.total_checkers()))
#         if s_no > 0 and s_no % 10000 == 0:
#           print(s_no, 'switches with p =', pos_p)
#     with open('XBS_ER_N100_p20_posp'+str(int(pos_p*100))+'.pkl', 'wb') as out_f:
#         pickle.dump(S.A, out_f)
# with open('XBS_ER_N100_p20.pkl', 'wb') as out_f:
#     pickle.dump(result, out_f)

with open('XBS_ER_N100_p20.pkl', 'rb') as in_f:
    result = pickle.load(in_f)

plt.figure(figsize=(8, 3))
cmap = colors.ListedColormap(['white', 'tab:blue'])
plt.subplot(1, 4, 1)
with open('XBS_ER_N100_p20_initial.pkl', 'rb') as in_f:
    A = pickle.load(in_f)
plt.imshow(A, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.subplot(1, 4, 2)
with open('XBS_ER_N100_p20_posp40.pkl', 'rb') as in_f:
    A = pickle.load(in_f)
plt.imshow(A, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.subplot(1, 4, 3)
with open('XBS_ER_N100_p20_posp70.pkl', 'rb') as in_f:
    A = pickle.load(in_f)
plt.imshow(A, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.subplot(1, 4, 4)
with open('XBS_ER_N100_p20_posp95.pkl', 'rb') as in_f:
    A = pickle.load(in_f)
plt.imshow(A, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('XBSconvergenceMats.pdf', format='pdf')
plt.show()


plt.figure(figsize=(6,5))
paltt = ['#277da1', '#f9c74f', '#f94144']
maxswitch=20000

plt.scatter([i[2]/10000 for i in result[0.4][:maxswitch]], [i[1] for i in result[0.4][:maxswitch]], c=paltt[0], s=1, alpha=.2, label=r'$p_{XBS}=0.4$')
r_sink = np.mean([i[1] for i in result[0.4][2000:]])
plt.plot([0, 30], [r_sink, r_sink], c=paltt[0], lw=1, ls='--')

plt.scatter([i[2]/10000 for i in result[0.7][:maxswitch]], [i[1] for i in result[0.7][:maxswitch]], c=paltt[1], s=1, alpha=.2, label=r'$p_{XBS}=0.7$')
r_sink = np.mean([i[1] for i in result[0.7][2000:]])
plt.plot([0, 30], [r_sink, r_sink], c=paltt[1], lw=1, ls='--')

plt.scatter([i[2]/10000 for i in result[0.95][:maxswitch]], [i[1] for i in result[0.95][:maxswitch]], c=paltt[2], s=1, alpha=.2, label=r'$p_{XBS}=0.95$')
r_sink = np.mean([i[1] for i in result[0.95][2000:]])
plt.plot([0, 30], [r_sink, r_sink], c=paltt[2], lw=1, ls='--')
#plt.plot([i[2]/10000 for i in result[0.7]], c='#FD8060')
#plt.plot([i[2]/10000 for i in result[0.4]], c='#EDC951')
plt.xlabel(r'Remaining switches')
plt.ylabel(r'Assortativity')
plt.text(30.2, -.03, r'$\times 10^4$')
#plotLabelY = 0.95*(plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0])+plt.gca().get_ylim()[0]
#plt.text(-6000, plotLabelY, 'b', weight='bold', fontsize=24)
plt.xlim([0, 30])
pltleg = plt.legend(frameon=False)
coloridx = 0
for text in pltleg.get_texts():
    text.set_color(paltt[coloridx])
    coloridx += 1
plt.savefig('XBSconvergence.pdf', format='pdf')
plt.show()




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
