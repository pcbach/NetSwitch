import numpy as np
import matplotlib.pyplot as plt
import igraph as ig

n1, n2 = 100, 100
d1, d2 = 3, 7
D = np.concatenate([np.repeat(d1, n1), np.repeat(d2, n2)])

if ig.is_graphical(D):
    G = ig.Graph.Degree_Sequence(D, method='vl')
else:
    raise 'D is not graphical!'
