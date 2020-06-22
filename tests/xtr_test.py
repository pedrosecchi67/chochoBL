import numpy as np
import matplotlib.pyplot as plt

from chochoBL import *

import pytest

'''
Module with tests to assert the position of transition in flat plate flow
'''

rho=defatm.rho
mu=defatm.mu

theta_over_dfs=0.66412
H_ideal=2.59109

Rex_crit=1e6

plotgraph=True
plotMoody=True

def test_trans():
    '''
    Assert the position of detected transition
    '''

    Lx=1.0
    Ly=1.0

    Re_target=5e5

    Uinf=Re_target*mu/(Lx*rho)

    msh, x0, q, xs, th_ideal, indmat=_gen_flatplate(Uinf=Uinf, echo=True, factor=1.0, Lx=Lx, Ly=Ly, nm=50, nn=2, Ncrit=6.0, A_transition=1.0, adj=True)

    solution, soln=msh.opt.solve(x0, q, solobj=True, method='CG', maxiter=500, relgtol=5e-3)

    if plotgraph:
        plt.scatter(xs, solution['th11'], label='numeric')
        plt.scatter(xs, th_ideal, label='analytic')

        plt.grid()
        plt.legend()

        plt.show()

        plt.scatter(xs, msh.gr.get_value('sigma_N')[0])

        plt.grid()
        plt.title('sigma_N')

        plt.show()

    if plotMoody:
        distJ=msh.dcell_dnode[1]

        Rex=rho*xs*Uinf/mu
        plt.scatter(xs, msh.gr.get_value('p')[0], label='numeric')
        # plt.scatter(Rex[Rex>0.0], 0.664/np.sqrt(Rex[Rex>0.0]), label='analytic')

        plt.grid()
        plt.legend()
        # plt.ylim((0.0, 1e-2))

        plt.show()

    print(solution, soln)

def _gen_flatplate(Uinf=1.0, echo=False, factor=1.2, Lx=1.0, Ly=1.0, nm=10, nn=2, Ncrit=6.0, A_transition=50.0, adj=False):
    '''
    Generate a flat plate with given discretization and dimensions+freestream conditions
    '''

    msh=mesh(Uinf=Uinf, echo=echo, A_transition=A_transition, Ncrit=Ncrit)

    xs=np.linspace(0.0, Lx, nm)
    ys=np.linspace(-Ly/2, Ly/2, nn)

    yy, xx=np.meshgrid(ys, xs)

    xaux=xx.reshape(nm*nn)

    for i in range(nm):
        for j in range(nn):
            msh.add_node([xx[i, j], yy[i, j], 0.0])

    indmat=np.arange(nm*nn, dtype='int').reshape((nm, nn))

    for i in range(nm-1):
        for j in range(nn-1):
            msh.add_cell({indmat[i, j], indmat[i+1, j], indmat[i+1, j+1], indmat[i, j+1]})

    normals=np.zeros((nm*nn, 3))
    normals[:, 2]=1.0

    msh.compose(normals, transition_CC=indmat[0, :] if adj else None)
    msh.graph_init()

    qx=Uinf*np.ones_like(xaux)

    qx[indmat[0, :]]=0.0

    delta_FS=np.zeros_like(qx)

    valid=qx>0.0

    delta_FS[valid]=np.sqrt(mu*xaux[valid]/(rho*qx[valid]))

    Rex=qx*xaux*rho/mu

    N=np.zeros_like(qx)

    N[Rex>Rex_crit]=7.0

    print(N)

    th=delta_FS*theta_over_dfs*factor
    H=np.ones_like(xaux)*H_ideal

    x0={'n':np.zeros_like(th), 'th11':th, 'H':H, 'beta':np.zeros_like(th), 'N':N}

    return msh, x0, {'qx':qx, 'qy':np.zeros_like(qx), 'qz':np.zeros_like(qx)}, xaux, th/factor, indmat
