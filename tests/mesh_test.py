import numpy as np
import time as tm

from chochoBL import *

import pytest

Lx=1.0
Ly=1.0

rho=defatm.rho
mu=defatm.mu

Uinf=1.0

Tau=0.27639
theta_over_dfs=0.55660
H_ideal=2.42161

alpha=0.1

gamma=np.pi/4

def test_meshgen():
    numR=_numeric(1.2)
    anR=_analytic(1.2)

    assert np.abs((numR-anR)/anR)<5e-2

def _numeric(factor):
    nm=100
    nn=50
    nnodes=nm*nn

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

    t=tm.time()

    msh.compose(normals)

    print('Mesh composing: ', tm.time()-t)

    qx=Uinf*xaux**alpha

    # to consider the coordinate system x axis direction inversion at the attachment line

    delta_FS=np.sqrt(mu*xaux/(rho*qx))

    th=delta_FS*theta_over_dfs*factor
    H=np.ones_like(xaux)*H_ideal

    t=tm.time()

    rmass, rmomx, rmomz, ren, rts=msh.mesh_getresiduals(np.zeros_like(th), th, H, \
        np.zeros_like(th), np.zeros_like(th), qx, np.zeros_like(qx), np.zeros_like(qx))

    print('Residual calculation: ', tm.time()-t)

    numR=np.sum(rmomx)

    return numR

def _analytic(factor):
    # analytic results

    nm=40
    nn=2
    nnodes=nm*nn

    xs=np.linspace(Lx, 0.0, nm, endpoint=False)
    xs=np.flip(xs)
    ys=np.linspace(-Ly/2, Ly/2, nn)

    yy, xx=np.meshgrid(ys, xs)

    qx=Uinf*xx**alpha

    # to consider the coordinate system x axis direction inversion at the attachment line

    delta_FS=np.sqrt(mu*xx/(rho*qx))

    th=delta_FS*theta_over_dfs*factor
    H=np.ones_like(xx)*H_ideal

    deltastar=H*th

    J=th*qx**2*rho
    M=deltastar*rho*qx

    dqx_dx=np.gradient(qx, xs, axis=0)

    tau=Tau*qx*mu/th

    r=np.gradient(J, xs, axis=0)+M*dqx_dx-tau

    Rtot=np.trapz(r[:, 0], x=xs)*Ly

    return Rtot
