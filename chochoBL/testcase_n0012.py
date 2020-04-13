from network import *
import numpy as np
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pickle
import time as tm

#test case including external velocity field from NACA 0012 airfoil as analysed by LovelacePM

data=open('n0012_qe_data', 'rb')
qedict=pickle.load(data)
qes=qedict['velmat']
pts=qedict['pts']
#transferring from LovelacePM's panel strip notation to network module's notation
pts.reverse()
for i in range(len(pts)):
    pts[i].reverse()
    qes[i].reverse()
t=tm.time()
ntw=network(mat=pts, velmat=qes, idisc=1000, jdisc=50, ndisc=20, thickness=0.07)#, delta_heuristics=lambda x, y : (x+1.0)/2)
for i in range(ntw.idisc-1):
    ntw.propagate(i)
ntw.calc_delta()
print(tm.time()-t, ' s')
#ntw.plot_delta(factor=1.0)
mid=floor(ntw.jdisc/2)
ntw.plot_ue()
plt.show()
for i in range(0, ntw.idisc-1, 50):
    plt.plot(ntw.us[i, mid, :], ntw.thicks[i, mid, :])
    plt.show()