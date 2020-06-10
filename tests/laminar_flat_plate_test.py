from chochoBL import *

import numpy as np
import numpy.linalg as lg
import scipy.optimize as sopt

import time as tm

import matplotlib.pyplot as plt

import pytest

# dimensions and definitions
nm=40
nn=2
nnodes=nm*nn

Lx=1.0
Ly=1.0

rho=defatm.rho
mu=defatm.mu

Uinf=2.0

Tau=0.36034
theta_over_dfs=0.29235
H_ideal=2.21622

def test_laminar_flat_plate():
    '''
    Define a laminar flat plate subject to normal flow (FS self-similar flow with alpha=1.0)
    and compare its total (integrated over area) residual to analytic results
    '''

    soln_num=sopt.root(_numeric, x0=0.9)
    soln_an=sopt.root(_analytic, x0=0.9)

    assert np.abs(soln_num.x-soln_an.x)<5e-2*np.abs(soln_an.x) # tolerating only up to 5% deviation from FS solution

def _numeric(factor):
    xs=np.linspace(Lx, 0.0, nm, endpoint=False)
    xs=np.flip(xs)
    ys=np.linspace(-Ly/2, Ly/2, nn)

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

    th=delta_FS*theta_over_dfs*factor
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

    msh.gr.calculate()

    numR=np.sum(msh.gr.get_value('Rmomx')[0])

    return numR

def _analytic(factor):
    # analytic results

    xs=np.linspace(Lx, 0.0, nm, endpoint=False)
    xs=np.flip(xs)
    ys=np.linspace(-Ly/2, Ly/2, nn)

    yy, xx=np.meshgrid(ys, xs)

    qx=xx*Uinf

    # to consider the coordinate system x axis direction inversion at the attachment line

    delta_FS=np.sqrt(mu*xx/(rho*qx))

    th=delta_FS*theta_over_dfs*factor
    H=np.ones_like(xx)*H_ideal

    deltastar=H*th

    J=th*qx**2*rho
    M=deltastar*rho*qx

    dqx_dx=Uinf

    tau=Tau*qx*mu/th

    r=np.gradient(J, xs, axis=0)+M*dqx_dx-tau

    Rtot=np.trapz(r[:, 0], x=xs)*Ly

    return Rtot
