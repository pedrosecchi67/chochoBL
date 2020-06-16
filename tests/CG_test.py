import numpy as np
import numpy.linalg as lg
import scipy.optimize as sopt

import matplotlib.pyplot as plt

import pytest

from chochoBL import *

from findiff_tests import total_residual

'''
Test script for asserting the capability of solving a boundary layer problem
via squared residual minimization and conjugate gradient method
'''

# dimensions and definitions
Lx=1.0
Ly=1.0

rho=defatm.rho
mu=defatm.mu

Uinf=2.0

Tau=0.36034
theta_over_dfs=0.29235
H_ideal=2.21622

plotgraph=False

def test_laminar_flat_plate():
    '''
    Define a laminar flat plate subject to normal flow (FS self-similar flow with alpha=1.0)
    and solve its boundary layer problem via gradient descent
    '''

    nm=20; nn=2
    msh, vals, Jxx_ideal=_getmesh(0.8, nm=nm, nn=nn)

    xs=np.flip(np.linspace(Lx, 0.0, nm, endpoint=False))
    ys=np.linspace(0.0, Ly, nn)

    yy, xx=np.meshgrid(ys, xs)

    xs=xx.reshape(np.size(xx))

    # definitions for gradient descent
    beta=2e-2

    niter=2000

    for i in range(niter):
        values, grads=msh.calculate_graph()

        for n in vals:
            for p in vals[n]:
                vals[n][p]-=beta*grads[p]

        msh.set_values(vals)

        ndev=lg.norm(vals['th11']['th11']*vals['q']['qx']**2*rho-Jxx_ideal)

        if i==0:
            ndev0=ndev

            if plotgraph:
                plt.scatter(xs, vals['th11']['th11']*vals['q']['qx']**2*rho, label='numeric')
                plt.scatter(xs, Jxx_ideal, label='ideal')

                plt.grid()
                plt.legend()

                plt.show()
        elif i==niter-1:
            ndevf=ndev

            if plotgraph:
                plt.scatter(xs, vals['th11']['th11']*vals['q']['qx']**2*rho, label='numeric')
                plt.scatter(xs, Jxx_ideal, label='ideal')

                plt.grid()
                plt.legend()

                plt.show()

        print(total_residual(values), total_residual(grads), ndev/ndev0)

    assert np.abs(ndevf/ndev0)<5e-2

def _getmesh(factor0, nm=20, nn=2):
    nnodes=nm*nn

    # numeric results
    xs=np.flip(np.linspace(Lx, 0.0, nm, endpoint=False))
    ys=np.linspace(0.0, Ly, nn)

    yy, xx=np.meshgrid(ys, xs)

    msh=mesh(Uinf=Uinf)

    n=0

    xaux=[]

    indmat=np.zeros((nm, nn), dtype='int')

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            msh.add_node([x, y, 0.0])
            xaux.append(x)

            indmat[i, j]=n
            n+=1

    for i in range(nm-1):
        for j in range(nn-1):
            msh.add_cell({indmat[i, j], indmat[i+1, j], indmat[i+1, j+1], indmat[i, j+1]})

    normals=np.zeros((nnodes, 3))
    normals[:, 2]=1.0

    xaux=np.array(xaux)

    msh.compose(normals)

    msh.graph_init()

    qx=Uinf*xaux

    delta_FS=np.sqrt(mu*xaux/(rho*qx))

    th=delta_FS*theta_over_dfs*factor0
    H=np.ones_like(xaux)*H_ideal

    vals={
        'q':{'qx':qx, 'qy':np.zeros_like(qx), 'qz':np.zeros_like(qx)},
        'th11':{'th11':th},
        'H':{'H':H},
        'n':{'n':np.zeros_like(qx)},
        'N':{'N':np.zeros_like(qx)},
        'beta':{'beta':np.zeros_like(qx)}
    }

    msh.set_values(vals)

    return msh, vals, th*qx**2*rho/factor0
