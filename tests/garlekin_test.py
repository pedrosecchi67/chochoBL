from chochoBL import *

import numpy as np
import scipy.sparse as sps
import time as tm
import random as rnd

import pytest

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
