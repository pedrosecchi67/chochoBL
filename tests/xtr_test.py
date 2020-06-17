import numpy as np

from chochoBL import *

import pytest

'''
Module with tests to assert the position of transition in flat plate flow
'''

rho=defatm.rho
mu=defatm.mu

theta_over_dfs=0.29235
H_ideal=2.21622

def _gen_flatplate(Uinf=1.0, Lx=1.0, Ly=1.0, nm=10, nn=2, Ncrit=6.0, A_transition=50.0):
    '''
    Generate a flat plate with given discretization and dimensions+freestream conditions
    '''

    msh=mesh(Uinf=Uinf, A_transition=A_transition, Ncrit=Ncrit)

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

    normals=np.zeros((nm, nn, 3))
    normals[:, :, 2]=1.0

    msh.compose(normals)

    qx=Uinf*np.ones_like(xaux)

    qx[indmat[0, :]]=0.0

    delta_FS=np.sqrt(mu*xaux/(rho*qx))

    th=delta_FS*theta_over_dfs
    H=np.ones_like(xaux)*H_ideal

    x0={'n':np.zeros_like(th), 'th11':th, 'H':H, 'beta':np.zeros_like(th), 'N':np.zeros_like(th)}

    return msh, x0, {'qx':qx, 'qy':np.zeros_like(qx), 'qz':np.zeros_like(qx)}
