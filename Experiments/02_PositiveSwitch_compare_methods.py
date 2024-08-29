from NetSwitchAlgs import *
import pickle
import random
import igraph as ig
import os.path
import matplotlib.pyplot as plt

n = 500
p = 0.05
swt_batch_count = 100

res_fname = "NetPosSwitchingResults/POS_ER_N"+str(int(n))+"_p"+str(int(p*100))+"_bch"+str(swt_batch_count)+".pkl"

if not os.path.isfile(res_fname):
    #RANDOM Switching
    random.seed(0)
    ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
    S = NetSwitch(ERgraph)
    rRand = [(S.swt_done, S.assortativity_coeff())]
    while True:
        ret = S.switch_A(alg='RAND', count=swt_batch_count)
        rRand.append((S.swt_done, S.assortativity_coeff()))
        if ret != -1:
            break
    #ARand = np.copy(S.A)
    print('RAND done')

    result = {'RAND': rRand, 'ORDR': [(0, 0)], 'ORDD': [(0, 0)], 'GRDY': [(0, 0)], 'XBS': [(0, 0)]}
    with open(res_fname, 'wb') as out_f:
        pickle.dump(result, out_f)

    random.seed(0)
    ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
    S = NetSwitch(ERgraph)
    rOrdr = [(S.swt_done, S.assortativity_coeff())]
    while True:
        ret = S.switch_A(alg='ORDR', count=swt_batch_count)
        rOrdr.append((S.swt_done, S.assortativity_coeff()))
        if ret != -1:
            break
    #ARowr = np.copy(S.A)
    print('ORDR done')

    result = {'RAND': rRand, 'ORDR': rOrdr, 'ORDD': [(0, 0)], 'GRDY': [(0, 0)], 'XBS': [(0, 0)]}
    with open(res_fname, 'wb') as out_f:
        pickle.dump(result, out_f)

    random.seed(0)
    ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
    S = NetSwitch(ERgraph)
    rOrdd = [(S.swt_done, S.assortativity_coeff())]
    while True:
        ret = S.switch_A(alg='ORDD', count=swt_batch_count)
        rOrdd.append((S.swt_done, S.assortativity_coeff()))
        if ret != -1:
            break
    #ADiag = np.copy(S.A)
    print('ORDD done')

    result = {'RAND': rRand, 'ORDR': rOrdr, 'ORDD': rOrdd, 'GRDY': [(0, 0)], 'XBS': [(0, 0)]}
    with open(res_fname, 'wb') as out_f:
        pickle.dump(result, out_f)

    random.seed(0)
    ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
    S = NetSwitch(ERgraph)
    rGrdy = [(S.swt_done, S.assortativity_coeff())]
    while True:
        ret = S.switch_A(alg='GRDY', count=swt_batch_count)
        rGrdy.append((S.swt_done, S.assortativity_coeff()))
        if ret != -1:
            break
    print('GRDY done')

    result = {'RAND': rRand, 'ORDR': rOrdr, 'ORDD': rOrdd, 'GRDY': rGrdy, 'XBS': [(0, 0)]}
    with open(res_fname, 'wb') as out_f:
        pickle.dump(result, out_f)

    random.seed(0)
    ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
    S = NetSwitch(ERgraph)
    rXbs = [(S.swt_done, S.assortativity_coeff())]
    while True:
        ret = S.XBS(pos_p=1.0, count=swt_batch_count)
        rXbs.append((S.swt_done, S.assortativity_coeff()))
        if ret != -1:
            break
    print('XBS done')

    result['XBS'] =  rXbs
    with open(res_fname, 'wb') as out_f:
        pickle.dump(result, out_f)

else:
    with open(res_fname, 'rb') as in_f:
        result = pickle.load(in_f)
    plt.plot([i[0] for i in result['RAND']], [i[1] for i in result['RAND']], label='Random')
    plt.plot([i[0] for i in result['ORDR']], [i[1] for i in result['ORDR']], label='Ordered (row)')
    plt.plot([i[0] for i in result['ORDD']], [i[1] for i in result['ORDD']], label='Ordered (diag)')
    plt.plot([i[0] for i in result['GRDY']], [i[1] for i in result['GRDY']], label='Greedy')
    plt.plot([i[0] for i in result['XBS']], [i[1] for i in result['XBS']], label=r'XBS ($p_{XBS}=1.0$)')
    plt.legend(frameon=False)
    plt.show()