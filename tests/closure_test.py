import numpy as np
import numpy.random as rnd
import scipy.sparse as sps

from chochoBL import *

import pytest

def _standard_data(n):
    Hk=np.interp(rnd.random(n), [0.0, 1.0], [2.0, 6.0])
    Me=np.interp(rnd.random(n), [0.0, 1.0], [0.1, 0.6])
    Reth=10.0**np.interp(rnd.random(n), [0.0, 1.0], [2.0, 8.0])
    return Reth, Me, Hk

def _standard_perturbations(*args, factor=1e-7):
    return tuple(a*factor for a in args)

def _arr_compare(a, b, tol=1e-5, relative=None):
    if relative is None:
        return np.all(np.abs(a-b)<tol)
    else:
        return np.all(np.abs((a-b)/relative)<tol)

def test_Hk():
    H=rnd.random(100)*2.0+2.0
    Me=rnd.random(100)*0.6+0.1

    pH=1e-7*H
    pMe=1e-7*Me

    dydx_num=(Hk(H+pH, Me)-Hk(H, Me))/pH

    dydx_an=np.diag(dHk_dH(H, Me).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

    dydx_num=(Hk(H, Me+pMe)-Hk(H, Me))/pMe

    dydx_an=np.diag(dHk_dMe(H, Me).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

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

def test_Cd_laminar():
    Reth_std, Me_std, Hk_std=_standard_data(100)
    pReth, pMe, pHk=_standard_perturbations(Reth_std, Me_std, Hk_std)

    dydx_num=(Cd_laminar(Reth_std, Hk_std+pHk)-Cd_laminar(Reth_std, Hk_std))/pHk

    dydx_an=np.diag(dCd_laminar_dHk(Reth_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

    dydx_num=(Cd_laminar(Reth_std+pReth, Hk_std)-Cd_laminar(Reth_std, Hk_std))/pReth

    dydx_an=np.diag(dCd_laminar_dReth(Reth_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

def test_Hstar_turbulent():
    Reth_std, Me_std, Hk_std=_standard_data(100)
    pReth, pMe, pHk=_standard_perturbations(Reth_std, Me_std, Hk_std)

    dydx_num=(Hstar_turbulent(Me_std+pMe, Hk_std)-Hstar_turbulent(Me_std, Hk_std))/pMe

    dydx_an=np.diag(dHstar_turbulent_dMe(Me_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

    dydx_num=(Hstar_turbulent(Me_std, Hk_std+pHk)-Hstar_turbulent(Me_std, Hk_std))/pHk

    dydx_an=np.diag(dHstar_turbulent_dHk(Me_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-2, \
        relative=dydx_an) #tolerance set higher since only three significative algorisms were considered

def test_Hprime_turbulent():
    Reth_std, Me_std, Hk_std=_standard_data(100)
    pReth, pMe, pHk=_standard_perturbations(Reth_std, Me_std, Hk_std)

    dydx_num=(Hprime_turbulent(Me_std+pMe, Hk_std)-Hprime_turbulent(Me_std, Hk_std))/pMe

    dydx_an=np.diag(dHprime_turbulent_dMe(Me_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

    dydx_num=(Hprime_turbulent(Me_std, Hk_std+pHk)-Hprime_turbulent(Me_std, Hk_std))/pHk

    dydx_an=np.diag(dHprime_turbulent_dHk(Me_std, Hk_std).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

def test_Cf_turbulent():
    Reth_std, Me_std, Hk_std=_standard_data(100)
    pReth, pMe, pHk=_standard_perturbations(Reth_std, Me_std, Hk_std)

    passive={'gamma':1.4}

    dydx_num=(Cf_turbulent(Reth_std, Me_std+pMe, Hk_std, passive)-\
        Cf_turbulent(Reth_std, Me_std, Hk_std, passive))/pMe

    dydx_an=np.diag(dCf_turbulent_dMe(Reth_std, Me_std, Hk_std, passive).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

    dydx_num=(Cf_turbulent(Reth_std+pReth, Me_std, Hk_std, passive)-\
        Cf_turbulent(Reth_std, Me_std, Hk_std, passive))/pReth

    dydx_an=np.diag(dCf_turbulent_dReth(Reth_std, Me_std, Hk_std, passive).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

    dydx_num=(Cf_turbulent(Reth_std, Me_std, Hk_std+pHk, passive)-\
        Cf_turbulent(Reth_std, Me_std, Hk_std, passive))/pHk

    dydx_an=np.diag(dCf_turbulent_dHk(Reth_std, Me_std, Hk_std, passive).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

def test_Cd_turbulent():
    Reth_std, Me_std, Hk_std=_standard_data(100)
    pReth, pMe, pHk=_standard_perturbations(Reth_std, Me_std, Hk_std)

    passive={'gamma':1.4}

    dydx_num=(Cd_turbulent(Reth_std, Me_std+pMe, Hk_std, passive)-\
        Cd_turbulent(Reth_std, Me_std, Hk_std, passive))/pMe

    dydx_an=np.diag(dCd_turbulent_dMe(Reth_std, Me_std, Hk_std, passive).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

    dydx_num=(Cd_turbulent(Reth_std+pReth, Me_std, Hk_std, passive)-\
        Cd_turbulent(Reth_std, Me_std, Hk_std, passive))/pReth

    dydx_an=np.diag(dCd_turbulent_dReth(Reth_std, Me_std, Hk_std, passive).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

    dydx_num=(Cd_turbulent(Reth_std, Me_std, Hk_std+pHk, passive)-\
        Cd_turbulent(Reth_std, Me_std, Hk_std, passive))/pHk

    dydx_an=np.diag(dCd_turbulent_dHk(Reth_std, Me_std, Hk_std, passive).todense())

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)
