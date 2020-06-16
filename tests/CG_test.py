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

plotgraph=True

def test_jac():
    '''
    Test function and Jacobian by finite difference test
    '''

    msh, x0, q, _=_getmesh(1.0, nm=10, nn=10)

    x0=msh.opt.pack(x0)

    msh.opt.q=q

    px=x0*1e-7

    f0=msh.opt.fun(x0)
    grad=msh.opt.jac(x0)

    f1=msh.opt.fun(x0+px)

    num, an=(f1-f0), grad@px

    assert np.abs((num-an)/an)<1e-3

def test_laminar_flat_plate():
    '''
    Define a laminar flat plate subject to normal flow (FS self-similar flow with alpha=1.0)
    and solve its boundary layer problem via gradient descent
    '''

    nm=30; nn=2
    msh, x0, q, J=_getmesh(0.8, nm=nm, nn=nn)

    xs=np.flip(np.linspace(Lx, 0.0, nm, endpoint=False))
    ys=np.linspace(0.0, Ly, nn)

    yy, xx=np.meshgrid(ys, xs)

    xs=xx.reshape(np.size(xx))

    solution, soln=msh.opt.solve(x0, q, solobj=True, options={'maxiter':200, 'gtol':1e-4}, method='CG')

    print(soln.nit)

    num, an=solution['th11']*rho*msh.opt.q['qx']**2, J

    if plotgraph:
        plt.scatter(xs, num, label='numeric')
        plt.scatter(xs, an, label='analytic')

        plt.grid()
        plt.legend()

        plt.show()

    assert np.amax(np.abs(num-an))<5e-2*np.amax(an) # ensuring maximum deviation of 5% in momentum deffect

def _getmesh(factor0, nm=20, nn=2):
    nnodes=nm*nn

    # numeric results
    xs=np.flip(np.linspace(Lx, 0.0, nm, endpoint=False))
    ys=np.linspace(0.0, Ly, nn)

    yy, xx=np.meshgrid(ys, xs)

    msh=mesh(Uinf=Uinf, echo=True)

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

    x0={'n':np.zeros_like(th), 'th11':th, 'H':H, 'beta':np.zeros_like(th), 'N':np.zeros_like(th)}

    J=th*rho*qx**2/factor0

    return msh, x0, {'qx':qx, 'qy':np.zeros_like(qx), 'qz':np.zeros_like(qx)}, J
