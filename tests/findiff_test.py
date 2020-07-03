import numpy as np
import scipy.sparse.linalg as splg

import time as tm
from tqdm import tqdm

from chochoBL import *

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

def test_forward():
    msh, th, h, qx=_getmesh()

    niter=100

    factor=1.1

    th_ideal=th.copy()

    th*=factor
    n=np.zeros_like(th)
    beta=np.zeros_like(th)
    nts=np.zeros_like(th)
    qy=np.zeros_like(qx)
    qz=np.zeros_like(qx)

    fdf=1e-9

    npp=np.ones_like(n)*fdf
    thp=th*fdf
    hp=h*fdf
    betap=beta*fdf
    ntsp=nts*fdf

    rmass, rmomx, rmomz, ren, rts=msh.mesh_getresiduals(n, th, h, beta, nts, qx, qy, qz)

    t=tm.time()

    rmassj, rmomxj, rmomzj, renj, rtsj=msh.mesh_getresiduals_jac(n, th, h, beta, nts, qx, qy, qz)

    print('Jacobian generation: ', tm.time()-t)

    t=tm.time()

    var_an=residual.jac_mult(rmassj[:, :, :, 0], msh.cellmatrix, npp)
    var_an+=residual.jac_mult(rmassj[:, :, :, 1], msh.cellmatrix, thp)
    var_an+=residual.jac_mult(rmassj[:, :, :, 2], msh.cellmatrix, hp)
    var_an+=residual.jac_mult(rmassj[:, :, :, 3], msh.cellmatrix, betap)
    var_an+=residual.jac_mult(rmassj[:, :, :, 4], msh.cellmatrix, ntsp)

    print('Direct AD per Jacobian evaluation: ', tm.time()-t)

    rmassp, rmomxp, rmomzp, renp, rtsp=msh.mesh_getresiduals(n+npp, th+thp, h+hp, beta+betap, nts+ntsp, qx, qy, qz)

    var_num=rmassp-rmass

    assert np.all(np.abs(var_an-var_num)<np.abs(var_an)*1e-3)

    t=tm.time()

    var_an=residual.jac_mult(rmomxj[:, :, :, 0], msh.cellmatrix, npp)
    var_an+=residual.jac_mult(rmomxj[:, :, :, 1], msh.cellmatrix, thp)
    var_an+=residual.jac_mult(rmomxj[:, :, :, 2], msh.cellmatrix, hp)
    var_an+=residual.jac_mult(rmomxj[:, :, :, 3], msh.cellmatrix, betap)
    var_an+=residual.jac_mult(rmomxj[:, :, :, 4], msh.cellmatrix, ntsp)

    print('Direct AD per Jacobian evaluation: ', tm.time()-t)

    rmassp, rmomxp, rmomzp, renp, rtsp=msh.mesh_getresiduals(n+npp, th+thp, h+hp, beta+betap, nts+ntsp, qx, qy, qz)

    var_num=rmomxp-rmomx

    assert np.all(np.abs(var_an-var_num)<np.abs(var_an)*1e-3)

    t=tm.time()

    rmassd, rmomxd, rmomzd, rend, rtsd=\
        msh.mesh_getresiduals_d(n, th, h, beta, nts, qx, qy, qz, \
            nd=npp, th11d=thp, betad=betap, hd=hp, ntsd=ntsp)

    print('Forward AD: ', tm.time()-t)
