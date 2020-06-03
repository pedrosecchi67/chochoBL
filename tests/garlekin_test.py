from chochoBL import *

from three_equation_test import std_mesh_fulldata

import numpy as np
import scipy.sparse as sps
import time as tm
import random as rnd

import pytest

def _arr_compare(a, b, tol=1e-5, min_div=1e-14, relative=None):
    if relative is None:
        return np.all(np.abs(a-b)<tol)
    else:
        return np.all(np.abs(a-b)/\
            np.array([max([min_div, np.abs(r)]) for r in relative])<=tol)

def test_add_set():
    l=[{0, 1, 2, 3}]
    assert add_set(l, {3, 4, 5, 6})==1, "add_set test failed to include new element"
    assert add_set(l, {3, 2, 1, 0})==0, "add_set test failed to identify already included"

def test_v_residual():
    #assembling inputs
    xs=np.array([0.0, 1.0, 1.0, 0.0])
    ys=np.array([0.0, 0.0, 1.0, 1.0])
    v=np.array([0.0, 0.0, 1.0, 0.0])
    Rmat=v_residual_matrix(xs, ys)
    assert np.abs(Rmat[2, :]@v-1.0/9)<1e-5, "v residual test failed"

    xs=np.array([0.0, 1.0, 1.0, 0.0])
    ys=np.array([0.0, 0.0, 2.0, 2.0])
    v=np.array([0.0, 0.0, 1.0, 1.0])
    Rmat=v_residual_matrix(xs, ys)
    assert np.abs(Rmat[3, :]@v-1.0/3)<1e-5, "v residual test failed"

def test_dx_residual():
    #assembling inputs
    xs=np.array([0.0, 1.0, 1.0, 0.0])
    ys=np.array([0.0, 0.0, 1.0, 1.0])
    v=np.array([0.0, 1.0, 1.0, 0.0])
    Rmat=dvdx_residual_matrix(xs, ys)
    assert np.abs((Rmat[1, :]+Rmat[2, :])@v-1.0/2)<1e-5, "dv_dx residual failed"

def test_dy_residual():
    xs=np.array([0.0, 2.0, 2.0, 0.0])
    ys=np.array([1.0, 1.0, 2.0, 2.0])
    v=np.array([0.0, 0.0, 1.0, 1.0])
    Rmat=dvdy_residual_matrix(xs, ys)
    assert np.abs((Rmat[2, :]+Rmat[3, :])@v-1.0)<1e-5, "dv_dy residual failed"

def test_udvdx():
    xs=np.array([0.0, 1.0, 1.0, 0.0])
    ys=np.array([0.0, 0.0, 1.0, 1.0])
    u=np.array([0.0, 1.0, 1.0, 0.0])
    v=np.array([0.0, 1.0, 1.0, 0.0])
    Rmat=udvdx_residual_matrix(xs, ys)
    assert np.abs(u@(Rmat[1, :]+Rmat[2, :])@v-1.0/3)<1e-5, "udv_dx residual failed"

def test_udvdy():
    xs=np.array([0.0, 2.0, 2.0, 0.0])
    ys=np.array([1.0, 1.0, 2.0, 2.0])
    u=np.array([0.0, 1.0, 1.0, 0.0])
    v=np.array([0.0, 0.0, 1.0, 1.0])
    Rmat=udvdy_residual_matrix(xs, ys)
    assert np.abs(u@(Rmat[2, :]+Rmat[3, :])@v-1.0/2)<1e-5, "udv_dx residual failed"

def test_Rv_mesh():
    # Test residual definition across mesh cells for a large mesh

    msh=std_mesh_fulldata()

    u=np.linspace(0.0, 10.0, msh.nnodes)

    J=msh.dcell_dnode_compose((u,))

    u_c=J@u

    RJ=msh.v_res_Jac

    u_theoretical=np.zeros_like(u_c)

    for i, c in enumerate(msh.cells):
        u_theoretical[4*i:4*(i+1)]=c.Rv@u_c[4*i:4*(i+1)]
    
    u_c=RJ@u_c

    assert _arr_compare(u_c, u_theoretical, tol=1e-3, relative=u_c)

    u=np.linspace(0.0, 10.0, msh.nnodes)
    v=np.linspace(10.0, 0.0, msh.nnodes)

    u_c=J@u
    v_c=J@v

    r_theoretical=np.zeros_like(u_c)

    for i, c in enumerate(msh.cells):
        r_theoretical[4*i:4*(i+1)]=(u_c[4*i:4*(i+1)]@c.Rudvdx)@v_c[4*i:4*(i+1)]

    r_c, Ju, Jv=Rudvdx_residual(u_c, v_c, msh)

    assert _arr_compare(r_c, r_theoretical, relative=r_c, tol=1e-3)
    assert _arr_compare(Ju@u_c, Jv@v_c, relative=r_c, tol=1e-3)

    r_theoretical=np.zeros_like(u_c)

    for i, c in enumerate(msh.cells):
        r_theoretical[4*i:4*(i+1)]=(u_c[4*i:4*(i+1)]@c.Rudvdz)@v_c[4*i:4*(i+1)]

    r_c, Ju, Jv=Rudvdz_residual(u_c, v_c, msh)

    assert _arr_compare(r_c, r_theoretical, relative=r_c, tol=1e-3)
    assert _arr_compare(Ju@u_c, Jv@v_c, relative=r_c, tol=1e-3)
