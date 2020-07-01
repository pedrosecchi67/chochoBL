from chochoBL import *

import numpy as np
import numpy.linalg as lg
import scipy.optimize as sopt

import time as tm

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import pytest

# dimensions and definitions
Lx=1.0
Ly=1.0

rho=defatm.rho
mu=defatm.mu

gamma=0.3 # rotation angle for the mesh

Uinf=2.0

Tau=0.36034
theta_over_dfs=0.29235
H_ideal=2.21622

alpha=1.0

def test_laminar_flat_plate():
    '''
    Define a laminar flat plate subject to normal flow (FS self-similar flow with alpha=1.0)
    and compare its total (integrated over area) residual to analytic results
    '''

    soln_num=sopt.root(_numeric, x0=0.8)

    assert np.abs(soln_num.x-1.0)<5e-2 # tolerating only up to 5% deviation from FS solution

def test_laminar_flat_plate_rotated():
    '''
    Define a laminar flat plate subject to normal flow (FS self-similar flow with alpha=1.0)
    and prove to solve it's x and z momentum problems, for an FS flow, scaling a crossflow angle for
    the self-similar boundary layer
    '''

    assert _numeric_rotated(np.array([1.0, 0.0]), tested=True) # test wether calculating equal values for 
    # symmetric properties (Jxx and Jzz, tau_x and tau_x...) and other analytic properties

    soln_num=sopt.root(_numeric_rotated, x0=np.array([1.0, 0.0]))

    assert np.abs(soln_num.x[0]-1.0)<5e-2, "X momentum problem failed for rotated plate"
    assert np.abs(soln_num.x[1])<5e-2, "Z momentum problem failed for rotated plate"

def _numeric(factor):
    nm=20
    nn=2
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

    msh.compose(normals)

    qx=Uinf*xaux**alpha

    # to consider the coordinate system x axis direction inversion at the attachment line

    delta_FS=np.sqrt(mu*xaux/(rho*qx))

    th=delta_FS*theta_over_dfs*factor
    H=np.ones_like(xaux)*H_ideal

    rmass, rmomx, rmomz, ren, rts=msh.mesh_getresiduals(np.zeros_like(th), th, H, \
        np.zeros_like(th), np.zeros_like(th), qx, np.zeros_like(qx), np.zeros_like(qx))

    msh.mesh_getresiduals_b(np.zeros_like(th), th, H, \
        np.zeros_like(th), np.zeros_like(th), qx, np.zeros_like(qx), np.zeros_like(qx))

    numR=np.sum(rmomx)

    print(factor, numR)

    return numR

def _numeric_rotated(factors, tested=False):
    nm=20
    nn=20
    nnodes=nm*nn

    # numeric results for rotated coordinate system for velocities
    xs=np.flip(np.linspace(Lx, 0.0, nm, endpoint=False))
    ys=np.linspace(0.0, Ly, nn)

    yy, xx=np.meshgrid(ys, xs)

    cg, sg=np.cos(gamma), np.sin(gamma)

    msh=mesh(Uinf=Uinf)

    n=0

    xaux=[]

    indmat=np.zeros((nm, nn), dtype='int')

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            msh.add_node([x, y, 0.0])
            xaux.append(x*cg+y*sg)

            indmat[i, j]=n
            n+=1

    for i in range(nm-1):
        for j in range(nn-1):
            msh.add_cell({indmat[i, j], indmat[i+1, j], indmat[i+1, j+1], indmat[i, j+1]})

    normals=np.zeros((nnodes, 3))
    normals[:, 2]=1.0

    xaux=np.array(xaux)

    msh.compose(normals)

    qx=Uinf*xaux**alpha

    # to consider the coordinate system x axis direction inversion at the attachment line

    delta_FS=np.sqrt(mu*xaux/(rho*qx))

    th=delta_FS*theta_over_dfs*factors[0]
    H=np.ones_like(xaux)*H_ideal

    betas=np.ones_like(th)*np.pi*factors[1]/9

    rmass, rmomx, rmomz, ren, rts=msh.mesh_getresiduals(np.zeros_like(th), th, H, \
        betas, np.zeros_like(th), qx, np.zeros_like(qx), np.zeros_like(qx))

    msh.mesh_getresiduals_b(np.zeros_like(th), th, H, \
        np.zeros_like(th), np.zeros_like(th), qx, np.zeros_like(qx), np.zeros_like(qx))

    numR=np.sum(rmomx**2)
    numRz=np.sum(rmomz**2)

    print(factors, numR, numRz, np.degrees(gamma))

    return numR, numRz
