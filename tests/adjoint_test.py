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

def test_adj_findiff():
    prop='th11'
    factor=1e-9

    value1, grad1, dNdp, N1=calc_adj({}, prop)
    value2, grad2, _, N2=calc_adj({prop:factor}, prop)

    var_an=np.sum(grad1[prop]*factor)
    var_num=(value2-value1)

    assert np.abs((var_an-var_num)/var_an)<1e-3

    var_an=(np.sum(dNdp, axis=1)*factor)
    var_an=var_an.reshape(np.size(var_an))
    var_num=(N2-N1)

    print(var_an, var_num)

    dev=np.amax(np.abs(var_an-var_num))
    devtol=np.amax(np.abs(var_an))*1e-3

    print(dev, devtol)

    assert dev<=devtol

def calc_adj(pert={}, prop='th11'):
    ts=[]

    nm=100
    nn=50

    Lx=1.0
    Ly=1.0

    Re_target=7e5

    Uinf=Re_target*mu/(Lx*rho)

    msh, x0, q, xs, _, indmat=_gen_flatplate(Uinf=Uinf, echo=True, factor=1.2, Lx=Lx, Ly=Ly, nm=nm, nn=nn, Ncrit=1.0, A_transition=1.0)

    for p in pert:
        x0[p]+=pert[p]

    N=x0['N']

    t=tm.time()

    msh.opt.q=q

    msh.opt.set_value(msh.opt.pack(x0))

    values, grads=msh.calculate_graph(ends=['RTS'])

    ts.append(tm.time()-t)

    # p=msh.gr.get_value('p')[0]
    # p=p*Uinf

    distJ=msh.dcell_dnode[1]

    inv=None

    t=tm.time()

    derivs_dir=msh.gr.get_derivs_direct('N', ends=['RTS'])

    ts.append(tm.time()-t)

    t=tm.time()

    A=distJ.T@derivs_dir['RTS']
    b=distJ.T@values['RTS']

    N, inv=solve_0CC(A, -b, indmat[0, :], inv=inv)

    ts.append(tm.time()-t)

    t=tm.time()

    msh.set_values({'N':N}, nodes=['N'], nodal=False, reset=False)

    values, grads=msh.calculate_graph()

    ts.append(tm.time()-t)

    t=tm.time()

    inv=None

    A=distJ.T@derivs_dir['RTS']
    b=grads['N']

    psi, inv=solve_0CC(A.T, -b, indmat[0, :], inv=inv)
    psi=distJ@psi

    ts.append(tm.time()-t)

    t=tm.time()

    derivs_rev=msh.gr.get_derivs_reverse(prop='RTS', ends=['RTS'])

    ts.append(tm.time()-t)

    print(ts, sum(ts))

    grads={p:grads[p]+(0.0 if derivs_rev[p] is None else psi@derivs_rev[p]) for p in ['n', 'th11', 'H', 'beta']}

    # testing application of contour conditions to inverse derivatives
    inv=None
    dNdp, _=solve_0CC(A, -(distJ.T@derivs_rev[prop]).todense(), indmat[0, :], inv=inv)

    print(msh.gr.get_time_report())

    return total_residual(values), grads, dNdp, N
