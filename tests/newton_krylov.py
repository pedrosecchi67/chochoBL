import numpy as np
import scipy.sparse.linalg as splg

import time as tm
from tqdm import tqdm

from chochoBL import *

'''
Test for verifying the feasibility of a Newton-Krylov solution
'''

Lx=1.0
Ly=1.0

rho=defatm.rho
mu=defatm.mu

Uinf=1.0

Tau=0.27639
theta_over_dfs=0.55660
H_ideal=2.42161

alpha=0.1

def _getmesh(CC=False):
    nm=100
    nn=50
    nnodes=nm*nn

    xs=np.linspace(Lx, 0.0, nm, endpoint=False)
    xs=np.flip(xs)
    ys=np.linspace(-Ly/2, Ly/2, nn)

    yy, xx=np.meshgrid(ys, xs)

    indmat=np.arange(0, nnodes, dtype='int').reshape((nm, nn))

    msh=mesh(Uinf=Uinf, transition_CC=indmat[0, :] if CC else None)

    xaux=[]

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            msh.add_node([x, y, 0.0])
            xaux.append(x)

    for i in range(nm-1):
        for j in range(nn-1):
            msh.add_cell({indmat[i, j], indmat[i+1, j], indmat[i+1, j+1], indmat[i, j+1]})

    normals=np.zeros((nnodes, 3))
    normals[:, 2]=1.0

    xaux=np.array(xaux)

    t=tm.time()

    msh.compose(normals)

    print('Mesh composing: ', tm.time()-t)

    qx=Uinf*xaux**alpha

    # to consider the coordinate system x axis direction inversion at the attachment line

    delta_FS=np.sqrt(mu*xaux/(rho*qx))

    th=delta_FS*theta_over_dfs
    H=np.ones_like(xaux)*H_ideal

    return msh, th, H, qx

def test_J():
    msh, th, h, qx=_getmesh()

    niter=100

    factor=1.2

    th_ideal=th.copy()

    th*=factor
    n=np.zeros_like(th)
    beta=np.zeros_like(th)
    nts=np.zeros_like(th)
    qy=np.zeros_like(qx)
    qz=np.zeros_like(qx)

    opt=msh.opt

    margin=0.2

    x=np.hstack([n, th, h, beta, nts])

    t=tm.time()

    linop=opt.get_jac_Krylov(x, qx, qy, qz)

    atol=np.amax(np.abs(opt.residuals))*margin

    solninfo=splg.lgmres(linop, -opt.residuals, x0=np.zeros(msh.nnodes*5), maxiter=3, atol=atol)

    xp=solninfo[0]
    info=solninfo[1]

    print('t, status: ', tm.time()-t, info)

    print(np.amax(np.abs(linop@xp+opt.residuals)), atol/margin)
