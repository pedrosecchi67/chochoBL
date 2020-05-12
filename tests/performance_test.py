import numpy as np
import time as tm
import random as rnd

from chochoBL import *

def performance_test(nstat=5000, ntest=1250, ncpus=4):
    t=tm.time()
    for i in range(ntest):
        delta=rnd.random()
        d2q_dx2=rnd.random()
        d2q_dxdz=rnd.random()
        beta=rnd.random()*0.2 #multiplying to mantain reasonable small-crossflow assumptions
        dbeta_dx=rnd.random()*0.1
        dbeta_dz=rnd.random()*0.1
        Lambda=rnd.random()*4.0-2.0
        Red=10.0**(4.0+4*rnd.random())
        dq_dx=-Lambda*defatm.mu/(defatm.rho*delta**2)
        dq_dz=np.tan(beta)*dq_dx
        qe=Red*defatm.mu/(delta*defatm.rho)
        stat=station(defclosure, delta=delta, dq_dx=dq_dx, dq_dz=dq_dz, d2q_dx2=d2q_dx2, d2q_dxdz=d2q_dxdz, qe=qe, Uinf=qe, \
            beta=beta, dbeta_dx=dbeta_dx, dbeta_dz=dbeta_dz)
        stat.calcpropag()
    tunit=(tm.time()-t)/ntest
    ttot=tunit*nstat
    ttot_mult=ttot/ncpus
    print('total: ', ttot)
    print('total with multiprocessing: ', ttot_mult)
    print('unit: ', tunit)
    return ttot, ttot_mult, tunit

performance_test()