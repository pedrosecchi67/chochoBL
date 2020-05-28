import numpy as np
import numpy.random as rnd
import scipy.sparse as sps

import pytest

from chochoBL import *

def _standard_data(n):
    Hk=np.interp(rnd.random(n), [0.0, 1.0], [2.0, 6.0])
    Me=np.interp(rnd.random(n), [0.0, 1.0], [0.1, 0.6])
    Reth=10.0**np.interp(rnd.random(n), [0.0, 1.0], [2.0, 6.0])
    th11=10.0**np.interp(rnd.random(n), [0.0, 1.0], [-4.0, -2.0])
    return Reth, Me, Hk, th11

def _standard_perturbations(*args, factor=1e-7, relative=True):
    if relative:
        return tuple(a*factor for a in args)
    else:
        return tuple(np.ones_like(a)*factor for a in args)

def _ifvalid_divide(a, b):
    return np.array([ax/bx if bx!=0.0 else 0.0 for ax, bx in zip(a, b)])

def _arr_compare(a, b, tol=1e-5, relative=None):
    if relative is None:
        return np.all(np.abs(a-b)<tol)
    else:
        return np.all(_ifvalid_divide(np.abs(a-b), np.abs(relative))<tol)

def test_Reth_crit():
    Reth_std, Me_std, Hk_std, th11_std=_standard_data(100)
    pReth, pMe, pHk, pth11=_standard_perturbations(Reth_std, Me_std, Hk_std, th11_std)

    dydx_num=(Reth_crit(Hk_std+pHk)-Reth_crit(Hk_std))/pHk

    dydx_an=dReth_crit_dHk(Hk_std)

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)
