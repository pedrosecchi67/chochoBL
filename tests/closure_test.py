import numpy as np
import numpy.random as rnd
import scipy.sparse as sps

from chochoBL import *

import pytest

def _standard_data(n):
    Hk=np.interp(rnd.random(n), [0.0, 1.0], [2.0, 4.0])
    Me=np.interp(rnd.random(n), [0.0, 1.0], [0.0, 0.6])
    Reth=10.0**np.interp(rnd.random(n), [0.0, 1.0], [2.0, 8.0])
    return Reth, Me, Hk

_standdev=1e-4
def _standard_perturbations(*args):
    return tuple(np.ones_like(a)*_standdev for a in args)

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

    assert _arr_compare(dHstar_dHk_an, dHstar_dHk_num, tol=1e-4, \
        relative=dHstar_dHk_an)
