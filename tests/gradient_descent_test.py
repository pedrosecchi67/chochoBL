import numpy as np
import time as tm
from tqdm import tqdm

from chochoBL import *

'''
Test for verifying the feasibility of a least squares solution via gradient descent
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

def _getmesh():
    nm=20
    nn=5
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

    th=delta_FS*theta_over_dfs
    H=np.ones_like(xaux)*H_ideal

    return msh, th, H, qx

def test_GD():
    msh, th, h, qx=_getmesh()

    niter=100

    factor=1.1

    step_alpha=2.5e-9

    th_ideal=th.copy()

    th*=factor
    n=np.zeros_like(th)
    beta=np.zeros_like(th)
    nts=np.zeros_like(th)
    qy=np.zeros_like(qx)
    qz=np.zeros_like(qx)

    for i in tqdm(range(niter)):
        rmass, rmomx, rmomz, ren, rts=msh.mesh_getresiduals(n, th, h, beta, nts, qx, qy, qz)
        nb, thb, hb, betab, ntsb, qxb, qyb, qzb=msh.mesh_getresiduals_b(n, th, h, beta, nts, qx, qy, qz, \
            rmass_b=rmass, rmomx_b=rmomx, rmomz_b=rmomz, ren_b=ren, rts_b=rts)

        n-=nb*step_alpha
        th-=thb*step_alpha
        h-=hb*step_alpha
        beta-=betab*step_alpha
        nts-=ntsb*step_alpha
        qx-=qxb*step_alpha
        qy-=qyb*step_alpha
        qz-=qzb*step_alpha

    assert np.abs(th_ideal[-1]-th[-1])/np.amax(np.abs(th_ideal))<5e-2
