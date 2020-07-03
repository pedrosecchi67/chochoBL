import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt

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

plotResults=True

def _getmesh(CC=False):
    nm=70
    nn=2
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

    return msh, th, H, qx, xaux

def test_J():
    msh, th, h, qx, _=_getmesh(CC=True)

    factor=1.2

    th_ideal=th.copy()

    th*=factor
    n=np.zeros_like(th)
    beta=np.zeros_like(th)
    nts=np.zeros_like(th)
    qy=np.zeros_like(qx)
    qz=np.zeros_like(qx)

    opt=msh.opt

    # margin=0.2

    x=opt.arrmap(np.hstack([n, th, h, beta, nts]))
    resx=opt.fun(x, qx, qy, qz, translate=True)

    t=tm.time()

    linop=opt.get_jac_Krylov(x, qx, qy, qz, translate=True, fulljac=True)

    xp=x*1e-8

    resxp=opt.fun(x+xp, qx, qy, qz, translate=True)

    var_num=resxp-resx

    var_an=linop@xp

    assert np.all(np.abs(var_an-var_num)<=np.abs(var_an)*1e-3)

    t=tm.time()

    r=resx.copy()

    # dx=splg.lgmres(linop, -r)[0]

    dx, inv=solve_0CC(linop, -r, msh.CC, method='analytic')

    print('Jacobian system solving: ', tm.time()-t)

def test_NR():
    msh, th, h, qx, xaux=_getmesh(CC=True)

    niter=3

    factor=1.5

    alpha=1.0

    th_ideal=th.copy()

    th*=factor
    n=np.zeros_like(th)
    beta=np.zeros_like(th)
    nts=np.zeros_like(th)
    qy=np.zeros_like(qx)
    qz=np.zeros_like(qx)

    opt=msh.opt

    x=opt.arrmap(np.hstack([n, th, h, beta, nts]))

    for i in (range(niter)):
        r=opt.fun(x, qx, qy, qz, translate=True)

        J=opt.get_jac_Krylov(x, qx, qy, qz, translate=True)

        print(lg.norm(r))

        try:
            dx, inv=solve_0CC(J, -r, msh.CC)
        except:
            dx=np.zeros_like(x)

        x+=alpha*dx

    numsoln=opt._fromx_extract(opt.arrunmap(x)[0], 'th11')

    if plotResults:
        plt.scatter(xaux, numsoln, label='Newton-Raphson')
        plt.scatter(xaux, th_ideal, label='Ideal')
        plt.scatter(xaux, th_ideal*factor, label='Initial guess')

        plt.ylim(0.0, 1.5*np.amax(th_ideal))
        plt.grid()
        plt.legend()

        plt.show()

    numdev=np.abs(numsoln[-1]-th_ideal[-1])
    devmax=np.abs(th_ideal[-1])*5e-2

    isaccurate=(numdev<=devmax)

    assert isaccurate
