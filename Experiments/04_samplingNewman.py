import numpy as np
import matplotlib.pyplot as plt
import igraph as ig

n1, n2 = 300, 150
d1, d2 = 2, 4
D = np.concatenate([np.repeat(d1, n1), np.repeat(d2, n2)])
mean_k = np.mean(D)
max_k = np.max(D)

q = np.array([(k+1) * (D==k+1).sum() / mean_k for k in range(1, max_k)])
q /= q.sum()
x = np.array([q[k]/(k+1)**1 for k in range(max_k-1)]) # make vector x and force it to decay faster than q
x /= x.sum() # Normalize the sum to 1

ee =  np.outer(x,q) + np.outer(q,x) - np.outer(x,x)
q_var = np.var(np.concatenate([np.repeat(1, int(q[0]*1000)), np.repeat(3, int(q[-1]*1000))]))

m = np.outer(q,q) - ee #np.array([[(q[i] - x[i])*(q[j] - x[j]) for i in range(max_k-1)] for j in range(max_k-1)])
m /= np.sum(np.outer(np.arange(1,max_k), np.arange(1,max_k))*m)
r_target = +1
e_mat = np.outer(q,q) + (r_target * q_var) * m

print(ee)
print(-(np.sum([x[i]*(i+1) for i in range(max_k-1)])-np.sum([q[i]*(i+1) for i in range(max_k-1)]))**2/q_var)
print((np.trace(ee)-np.sum(ee @ ee))/(1-np.sum(ee @ ee)))

print(e_mat)
print((np.trace(e_mat)-np.sum(e_mat @ e_mat))/(1-np.sum(e_mat @ e_mat)))

if ig.is_graphical(D):
    G = ig.Graph.Degree_Sequence(D, method='vl')
else:
    raise 'D is not graphical!'
