import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle


filename = 'configs_custom_n9_m9.pkl'
with open(filename, 'rb') as in_f:
    data = pickle.load(in_f)

filename = 'configs_custom_n9_m9_noDB.pkl'
with open(filename, 'rb') as in_f:
    dataN = pickle.load(in_f)

all_configs_r = data['c']
sorted_config_idxs = np.argsort(all_configs_r)
print(np.array(all_configs_r)[sorted_config_idxs]/np.sum(all_configs_r))

#print(np.array(data['c'])[sorted_config_idxs])


# Create a figure
fig = plt.figure(figsize=(9, 3))  # 3 subplots wide

# Create grid spec with 3 columns and 2 rows, middle column has 2 subplots
gs = gridspec.GridSpec(2, 3, height_ratios=[3, 2], width_ratios=[.8, 1, .7])

# Adjust layout: leave space for ticks and labels
gs.update(wspace=0.3, hspace=0.11, left=0.07, right=0.995, top=0.95, bottom=0.15)

mcolors=['#560bad', '#f72585', '#4cc9f0']
# Left subplot (full height)
ax1 = plt.subplot(gs[:, 0])
ax1.hist([data['r'], data['s'][10240000]['r'], dataN['s'][10240000]['r']] , bins=8, density=True, color=[mcolors[0], mcolors[1], mcolors[2]], label=['Actual', 'Detailed Balance', 'No Rejection'])
ax1.set_xlim([-.85, +.65])
ax1.set_yticks([0, 0.5, 1, 1.5])
#plt.plot(np.linspace(0, .4, 100), normal(np.linspace(0, .4, 100), target_r, sigma))
ax1.set_xlabel('Degree Assortativity')
ax1.set_ylabel('Probability density')
leg1 = ax1.legend(frameon=False, fontsize=8, handlelength=0)
for text, color in zip(leg1.get_texts(), mcolors):
    text.set_color(color)
    text.set_fontweight('bold')

# Middle subplots (stacked vertically)
ax2a = plt.subplot(gs[0, 1], sharex=None)
ax2b = plt.subplot(gs[1, 1], sharex=ax2a)  # share x with top

colors=['#4cc9f0', '#e9c46a', '#2a9d8f', '#e76f51']
#np.array(data['c'])[sorted_config_idxs],
ax2a.plot(np.array(data['s'][320000]['e'])[sorted_config_idxs]/32, c=colors[2], marker='o', markersize=3, lw=0, alpha=1, label=r'$320$ k')
ax2a.plot(np.array(data['s'][2560000]['e'])[sorted_config_idxs]/256, c=colors[1], marker='o', markersize=3, lw=0, alpha=1, label=r'$2.56$ M')
ax2a.plot(np.array(data['s'][20480000]['e'])[sorted_config_idxs]/2048, c=colors[3], marker='o', markersize=3, lw=0, alpha=1, label=r'$20.48$ M')
ax2a.plot([0, len(data['e'])], [10000.0/len(data['e']), 10000.0/len(data['e'])], c='k', lw=2, ls='--', label=r'$1/|\mathcal{G}|$')
ax2a.text(0.01, 0.95, r'Detailed Balance', transform=ax2a.transAxes, fontsize=9, fontweight='bold', va='top', ha='left')
ax2a.set_ylim([1, 4])
ax2a.set_xlim([1, len(data['e'])])
ax2a.set_yticks([1, 2, 3, 4])
ax2a.legend(frameon=False, ncol=2, fontsize=8)
ax2b.set_xlabel('Graph configuration')
ax2b.set_ylabel(r'Sampling probability ($\times 10^{-4}$)')
ax2b.yaxis.set_label_coords(-0.08, 1.30)

ax2b.plot(np.array(dataN['s'][320000]['e'])[sorted_config_idxs]/32, c=colors[2], marker='o', markersize=3, lw=0, alpha=0.7, label=r'$8\times 10^4$ samples')
ax2b.plot(np.array(dataN['s'][2560000]['e'])[sorted_config_idxs]/256, c=colors[1], marker='o', markersize=3, lw=0, alpha=0.7, label=r'$32\times 10^4$ samples')
ax2b.plot(np.array(dataN['s'][20480000]['e'])[sorted_config_idxs]/2048, c=colors[3], marker='o', markersize=3, lw=0, alpha=0.7, label=r'$256\times 10^4$ samples')
ax2b.plot([0, len(data['e'])], [10000.0/len(data['e']), 10000.0/len(data['e'])], c='k', lw=2, ls='--', label=r'$1/|\mathcal{G}|$')
#ax2b.plot(10000*np.array(all_configs_r)[sorted_config_idxs]/np.sum(all_configs_r))
ax2b.text(0.01, 0.95, r'No Rejection', transform=ax2b.transAxes, fontsize=9, fontweight='bold', va='top', ha='left')
ax2b.set_xlim([1, len(data['e'])])
ax2b.set_yticks([1, 2, 3])
ax2b.set_ylim([1, 3])




# Right subplot (full height)
ax3 = plt.subplot(gs[:, 2])

ax3.plot([2**i*10000 for i in range(12)], [np.var(np.array(data['s'][2**i*10000]['e'])/(2**i*10000)) for i in range(12)], c=mcolors[1], lw=3, label='Detailed Balance')
ax3.plot([2**i*10000 for i in range(12)], [np.var(np.array(dataN['s'][2**i*10000]['e'])/(2**i*10000)) for i in range(12)], c=mcolors[2], lw=3, label='No Rejection')
ax3.set_xlabel('Sample count')
ax3.set_ylabel('Frequency variance')
ax3.yaxis.set_label_coords(-0.23, .5)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.legend(frameon=False, fontsize=9)

# Optional: remove ticks on shared axes
plt.setp(ax2a.get_xticklabels(), visible=False)

plt.show()