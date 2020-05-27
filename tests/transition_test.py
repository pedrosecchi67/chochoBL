import numpy as np
import numpy.random as rnd
import scipy.sparse as sps

import pytest

from chochoBL import *

def _standard_data(n):
    Hk=np.interp(rnd.random(n), [0.0, 1.0], [2.0, 6.0])
    Me=np.interp(rnd.random(n), [0.0, 1.0], [0.1, 0.6])
    Reth=10.0**np.interp(rnd.random(n), [0.0, 1.0], [2.0, 8.0])
    th11=10.0**np.interp(rnd.random(n), [0.0, 1.0], [-4.0, -2.0])
    return Reth, Me, Hk, th11

def _standard_perturbations(*args, factor=1e-7):
    return tuple(a*factor for a in args)

def _arr_compare(a, b, tol=1e-5, relative=None):
    if relative is None:
        return np.all(np.abs(a-b)<tol)
    else:
        return np.all(np.abs((a-b)/relative)<tol)

def test_dN_dReth():
    Reth_std, Me_std, Hk_std, th11_std=_standard_data(100)
    pReth, pMe, pHk, pth11=_standard_perturbations(Reth_std, Me_std, Hk_std, th11_std)

    dydx_num=(dN_dReth(Hk_std+pHk)-dN_dReth(Hk_std))/pHk

    dydx_an=d2N_dHkdReth(Hk_std)

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

def test_p():
    Reth_std, Me_std, Hk_std, th11_std=_standard_data(100)
    pReth, pMe, pHk, pth11=_standard_perturbations(Reth_std, Me_std, Hk_std, th11_std)

    dydx_num=(p(Hk_std+pHk, th11_std)-p(Hk_std, th11_std))/pHk

    dydx_an=np.diag(dp_dHk(Hk_std, th11_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)
