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

def test_f_crossflow():
    nsamples=1000
    factor=1e-7

    beta=(rnd.random(nsamples)-0.5)*np.pi/2
    Cf=rnd.random(nsamples)*0.01
    Me=rnd.random(nsamples)*0.5

    cosb=np.cos(beta)
    tb=np.tan(beta)

    f=f_crossflow(Cf, cosb, Me)

    df_dCf=df_crossflow_dCf(f, Cf)
    df_dbeta=df_crossflow_dbeta(f, tb)
    df_dMe=df_crossflow_dMe(f, Me)

    pbeta=factor*np.amax(beta)
    pCf=factor*np.amax(Cf)
    pMe=factor*np.amax(Me)

    pcosb=-np.sin(beta)*pbeta

    f2=f_crossflow(Cf+pCf, cosb, Me)

    dydx_num=(f2-f)/pCf
    dydx_an=df_dCf

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)
    
    f2=f_crossflow(Cf, cosb+pcosb, Me)

    dydx_num=(f2-f)/pbeta
    dydx_an=df_dbeta

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)
    
    f2=f_crossflow(Cf, cosb, Me+pMe)

    dydx_num=(f2-f)/pMe
    dydx_an=df_dMe

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)

def test_g_crossflow():
    nsamples=100
    factor=1e-7

    f=rnd.random(nsamples)*0.1-0.05

    pf=np.amax(f)*factor

    g=g_crossflow(f)
    g2=g_crossflow(f+pf)
    dg_df=dg_crossflow_df(f)

    dydx_num=(g2-g)/pf
    dydx_an=dg_df

    assert _arr_compare(dydx_an, dydx_num, tol=1e-3, \
        relative=dydx_an)
