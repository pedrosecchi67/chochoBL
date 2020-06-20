import numpy as np
import numpy.linalg as lg
import scipy.optimize as sopt
import scipy.sparse.linalg as slg
import scipy.sparse as sps

import time as tm

import pytest

'''
Test module to validate adjoint solution of transition problem
'''

from chochoBL import *

from xtr_test import _gen_flatplate

rho=defatm.rho
mu=defatm.mu

theta_over_dfs=0.66412
H_ideal=2.59109

Rex_crit=1e6

def test_syssolve():
    nm=50
    nn=100

    Lx=1.0
    Ly=1.0

    Re_target=5e5

    Uinf=Re_target*mu/(Lx*rho)

    msh, x0, q, xs, _, indmat=_gen_flatplate(Uinf=Uinf, echo=True, factor=1.0, Lx=Lx, Ly=Ly, nm=nm, nn=nn, Ncrit=6, A_transition=2.0)

    u=q['qx']
    N=x0['N']

    msh.opt.q=q

    msh.opt.set_value(msh.opt.pack(x0))

    values, grads=msh.calculate_graph()

    p=msh.gr.get_value('p')[0]
    p=p*Uinf

    distJ=msh.dcell_dnode[1]

    w=np.ones_like(u)*Uinf

    _, dRdu, dRudv=Rudvdx_residual(distJ@u, distJ@N, msh)
    _, dRdw, dRwdv=Rudvdx_residual(distJ@w, distJ@N, msh)

    dRudv=distJ.T@dRudv@distJ
    dRwdv=distJ.T@dRwdv@distJ

    A=dRudv+dRwdv
    b=distJ.T@msh.v_res_Jac@distJ@p

    inv=None

    t=tm.time()

    N, inv=solve_0CC(A, b, indmat[0, :], inv=inv)

    print(tm.time()-t)

    t=tm.time()

    N=solve_0CC(A, b, indmat[0, :], method='CG', inv=inv)

    print(tm.time()-t)

    plt.scatter(xs, N)
    plt.show()
