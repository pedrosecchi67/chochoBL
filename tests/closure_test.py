import numpy as np
import numpy.random as rnd
import scipy.sparse as sps

from chochoBL import *

import pytest

def _standard_data(n):
    Hk=np.interp(rnd.random(n), [0.0, 1.0], [2.0, 6.0])
    Me=np.interp(rnd.random(n), [0.0, 1.0], [0.0, 0.6])
    Reth=10.0**np.interp(rnd.random(n), [0.0, 1.0], [2.0, 8.0])
    return Reth, Me, Hk

def _standard_perturbations(*args, factor=1e-7):
    return tuple(a*factor for a in args)

def _arr_compare(a, b, tol=1e-5, relative=None):
    if relative is None:
        return np.all(np.abs(a-b)<tol)
    else:
        return np.all(np.abs((a-b)/relative)<tol)

def test_Hstar_laminar():
    Reth_std, Me_std, Hk_std=_standard_data(100)
    pReth, pMe, pHk=_standard_perturbations(Reth_std, Me_std, Hk_std)

    dHstar_dHk_num=(Hstar_laminar(Hk_std+pHk)-Hstar_laminar(Hk_std))/pHk

    dHstar_dHk_an=np.diag(dHstar_laminar_dHk(Hk_std).todense())

    assert _arr_compare(dHstar_dHk_an, dHstar_dHk_num, tol=1e-3, \
        relative=dHstar_dHk_an)

def test_Hprime_laminar():
    Reth_std, Me_std, Hk_std=_standard_data(100)
    pReth, pMe, pHk=_standard_perturbations(Reth_std, Me_std, Hk_std)

    dydx_num=(Hprime_laminar(Me_std, Hk_std+pHk)-Hprime_laminar(Me_std, Hk_std))/pHk

    dydx_an=np.diag(dHprime_laminar_dHk(Me_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

    dydx_num=(Hprime_laminar(Me_std+pMe, Hk_std)-Hprime_laminar(Me_std, Hk_std))/pMe

    dydx_an=np.diag(dHprime_laminar_dMe(Me_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

def test_Cf_laminar():
    Reth_std, Me_std, Hk_std=_standard_data(100)
    pReth, pMe, pHk=_standard_perturbations(Reth_std, Me_std, Hk_std)

    dydx_num=(Cf_laminar(Reth_std, Hk_std+pHk)-Cf_laminar(Reth_std, Hk_std))/pHk

    dydx_an=np.diag(dCf_laminar_dHk(Reth_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

    dydx_num=(Cf_laminar(Reth_std+pReth, Hk_std)-Cf_laminar(Reth_std, Hk_std))/pReth

    dydx_an=np.diag(dCf_laminar_dReth(Reth_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

def test_Cd_laminar_Hk():
    Reth_std, Me_std, Hk_std=_standard_data(100)
    pReth, pMe, pHk=_standard_perturbations(Reth_std, Me_std, Hk_std)

    dydx_num=(Cd_laminar(Reth_std, Hk_std+pHk)-Cd_laminar(Reth_std, Hk_std))/pHk

    dydx_an=np.diag(dCd_laminar_dHk(Reth_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

def test_Cd_laminar_Reth():
    Reth_std, Me_std, Hk_std=_standard_data(100)
    pReth, pMe, pHk=_standard_perturbations(Reth_std, Me_std, Hk_std)

    dydx_num=(Cd_laminar(Reth_std+pReth, Hk_std)-Cd_laminar(Reth_std, Hk_std))/pReth

    dydx_an=np.diag(dCd_laminar_dReth(Reth_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)
