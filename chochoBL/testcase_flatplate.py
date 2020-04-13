from network import *
import numpy as np
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time as tm

#test case for constructors: flat plate
ps=[[np.array([x, y, 0.0]) for x in np.linspace(0.0, 1.0, 100)] for y in np.linspace(0.0, 1.0, 50)]
qes=[[np.array([1.0+u/5, 0.0, 0.0]) for u in np.linspace(0.0, 1.0, 100)] for v in np.linspace(0.0, 1.0, 50)]
t=tm.time()
ntw=network(mat=ps, velmat=qes, idisc=500, jdisc=50, ndisc=50, thickness=0.06, delta_heuristics=lambda x, y : (x+1.0)/2)#, nstrategy=lambda x: x)
for i in range(ntw.idisc-1):
    ntw.propagate(i)
ntw.calc_delta()
print(tm.time()-t, ' s')
ntw.plot_delta(factor=100.0)
plt.plot(ntw.origin[ntw.attached[:, 0], 0, 0], ntw.delta[ntw.attached[:, 0], 0])
plt.show()
for i in range(0, ntw.idisc, 20):
    plt.plot(ntw.us[i, 0, :], ntw.thicks[i, 0, :])
    plt.show()