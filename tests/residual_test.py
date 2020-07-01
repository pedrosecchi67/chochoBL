import numpy as np

from chochoBL import residual, defatm

import pytest

'''
Script for testing Fortran routines in residual calculation
'''

def evalp(ksi, eta, p):
    return np.sum(np.array([[p[4*j+i]*ksi**i*eta**j for i in range(4)] for j in range(4)]))

def intp(p):
    s=0.0

    for i in range(4):
        for j in range(4):
            s+=p[4*j+i]/((i+1)*(j+1))

    return s

testpts=[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), \
    (0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]

testpols=[
    np.array([0.1, 2.0, -0.5, 1.0]),
    np.array([-0.5, 1.0, -3.0, 2.0]),
    np.array([-2.0, 1.0, -1.0, 2.0])
]

def test_mult():
    for p in testpols:
        for q in testpols:
            pp=residual.getpoly(p)
            pq=residual.getpoly(q)

            pmult=residual.getmult(pp, pq)

            for (ksi, eta) in testpts:
                assert np.abs(evalp(ksi, eta, pmult)-evalp(ksi, eta, pp)*evalp(ksi, eta, pq))<1e-5

def test_int01():
    for p in testpols:
        pp=residual.getpoly(p)

        assert np.abs(intp(pp)-residual.getint01(pp))<1e-5

def test_surfint():
    xs=np.array([0.0, 1.0, 1.0, 0.0])
    ys=np.array([0.0, 0.0, 1.0, 1.0])

    for p in testpols:
        pp=residual.getpoly(p)

        ideal=intp(pp)

        assert np.abs(residual.surfint(xs, ys, pp)-ideal)<1e-5

        dppx=residual.getder_ksi(pp)
        dppy=residual.getder_eta(pp)

        idealx=intp(dppx)
        idealy=intp(dppy)

        assert np.abs(residual.surfint_dx(xs, ys, pp)-idealx)<1e-5
        assert np.abs(residual.surfint_dy(xs, ys, pp)-idealy)<1e-5

def test_usurfint():
    xs=np.array([0.0, 1.0, 1.0, 0.0])
    ys=np.array([0.0, 0.0, 1.0, 1.0])

    for p in testpols:
        for q in testpols:
            pp=residual.getpoly(p)
            pq=residual.getpoly(q)

            dppx=residual.getder_ksi(pp)
            dppy=residual.getder_eta(pp)

            idealx=intp(residual.getmult(pq, dppx))
            idealy=intp(residual.getmult(pq, dppy))

            assert np.abs(residual.surfint_udx(xs, ys, pq, pp)-idealx)<1e-5
            assert np.abs(residual.surfint_udy(xs, ys, pq, pp)-idealy)<1e-5

def test_uw():
    qx=np.array([1.0, 1.0, 1.0, 1.0])
    qy=np.array([1.0, -1.0, -1.0, 1.0])
    qz=np.array([1.0, 1.0, 1.0, 1.0])

    mtosys=np.eye(3)

    mtosys[2, :], mtosys[1, :]=mtosys[1, :], mtosys[2, :]

    mtosys[1, :]=np.cross(mtosys[2, :], mtosys[0, :])

    u, w, q=residual.uwq(qx, qy, qz, mtosys)

    assert np.all(np.abs(qx-u)<1e-5)
    assert np.all(np.abs(qy-w)<1e-5)

    assert np.all(np.abs(q-np.sqrt(3))<1e-5)

    v_sonic=10.0

    mach=residual.mache(q, v_sonic)

    rho0=1.0
    uinf=2.0

    assert np.all(np.abs(mach-q/v_sonic)<1e-5)
    
    rhoe=(1.0-mach**2*(mach*v_sonic-uinf)/uinf)*rho0

    assert np.all(np.abs(rhoe-residual.rhoe(mach, v_sonic, rho0, uinf))<1e-5)

    th11=np.ones_like(qx)*1e-3

    mu=1e-5

    rethideal=rhoe*th11*q/mu

    assert np.all(np.abs(rethideal-residual.reth(q, rhoe, th11, mu))<1e-5)

def test_sigma():
    vals=np.array([-710.0, 70.0, -2.0, 0.0, 2.0, 70.0, 710.0])

    vals=residual.expit(vals)

    assert vals[3]==0.5
    assert vals[0]==0.0
    assert vals[6]==1.0

_garlekin_weights=[residual.n1, residual.n2, residual.n3, residual.n4]

def _udvdx_residual_squaremsh(p, q):
    xs=np.array([0.0, 1.0, 1.0, 0.0])
    ys=np.array([0.0, 0.0, 1.0, 1.0])

    pp=residual.getpoly(p)
    qq=residual.getpoly(q)

    dqq=residual.getder_ksi(qq)

    return np.array([intp(residual.getmult(n, residual.getmult(pp, dqq))) for n in _garlekin_weights])

def _udvdy_residual_squaremsh(p, q):
    xs=np.array([0.0, 1.0, 1.0, 0.0])
    ys=np.array([0.0, 0.0, 1.0, 1.0])

    pp=residual.getpoly(p)
    qq=residual.getpoly(q)

    dqq=residual.getder_eta(qq)

    return np.array([intp(residual.getmult(n, residual.getmult(pp, dqq))) for n in _garlekin_weights])

def test_matrixes():
    xs=np.array([0.0, 1.0, 1.0, 0.0])
    ys=np.array([0.0, 0.0, 1.0, 1.0])

    rvj=residual.get_rv_matrix(xs, ys)
    rdxj=residual.get_rdvdx_matrix(xs, ys)
    rdyj=residual.get_rdvdy_matrix(xs, ys)
    rudxj=residual.get_rudvdx_matrix(xs, ys)
    rudyj=residual.get_rudvdy_matrix(xs, ys)

    for p in testpols:
        pp=residual.getpoly(p)

        assert np.abs(intp(residual.getmult(pp, residual.n1))-rvj[0, :]@p)<1e-5
        assert np.abs(intp(residual.getmult(residual.getder_ksi(pp), residual.n1))-rdxj[0, :]@p)<1e-5
        assert np.abs(intp(residual.getmult(residual.getder_eta(pp), residual.n1))-rdyj[0, :]@p)<1e-5

        for q in testpols:
            assert np.all(np.abs(_udvdx_residual_squaremsh(p, q)-p@rudxj@q)<1e-5)
            assert np.all(np.abs(_udvdy_residual_squaremsh(p, q)-p@rudyj@q)<1e-5)

            qq=residual.getpoly(q)

            buff1=np.zeros_like(p)
            buff2=np.zeros_like(q)

            residual.matbyvec(rvj, p, buff1)

            assert np.all(np.abs(rvj@p-buff1)<1e-5)

            residual.mat3byvec(rudxj, p, q, buff2)

            assert np.all(np.abs(p@rudxj@q-buff2)<1e-5)

def _arr_compare(a, b, tol=1e-5, min_div=1e-14, relative=None):
    if relative is None:
        return np.all(np.abs(a-b)<tol)
    else:
        return np.all(np.abs(a-b)/\
            np.array([max([min_div, np.abs(r)]) for r in relative])<=tol)

from scipy.special import expit

def _m(Hk):
    return (0.058*(Hk-4.0)**2/(Hk-1.0)-0.068)/_l(Hk)

def _l(Hk):
    return (6.54*Hk-14.07)/Hk**2

def dN_dReth(Rethc, sg, Reth, Hk, A, ismult=True):

    return 0.01*np.sqrt((2.4*Hk-3.7+2.5*np.tanh(1.5*Hk-4.65))**2+0.15)*(sg if ismult else 1.0)

def p(dN_dR, th11, m, l):

    return dN_dR*((m+1.0)/2)*l/th11

def _A(Hk):
    return (1.415/(Hk-1.0)-0.489)*np.tanh(20.0/(Hk-1.0)-12.9)

def _B(Hk):
    return 3.295/(Hk-1.0)+0.44

def _log10Reth_crit(Hk):
    return (_A(Hk)+_B(Hk))

def Reth_crit(Hk):
    return 10.0**_log10Reth_crit(Hk)

_th11_tolerance=1e-8

def pfunc(Reth, Hk, th11, A):
    Rethc=Reth_crit(Hk)

    sg=expit((Reth-Rethc)*A)

    dNdR=dN_dReth(Rethc, sg, Reth, Hk, A, ismult=False)

    m=_m(Hk)

    l=_l(Hk)

    dNdR*=sg

    th11_aux=th11.copy()
    th11_aux[th11_aux<_th11_tolerance]=_th11_tolerance

    pval=p(dNdR, th11_aux, m, l)

    return pval

def test_quad_mesh():
    posits=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
    )

    xs=posits[:, 0]
    ys=posits[:, 1]

    mtosys=np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0]
        ]
    )

    qes=np.array(
        [1.0, 3.0, 1.0, 2.0]
    )

    rho=defatm.rho
    mu=defatm.mu
    a=defatm.v_sonic

    Hs=np.array(
        [2.5, 3.0, 3.0, 2.7]
    )

    thetas=np.array(
        [1e-3, 2e-3, 1e-3, 3e-3]
    )

    Nts=np.array([0.2, 0.1, 0.3, 0.4])

    betas=np.zeros_like(thetas)

    deltastars=Hs*thetas

    J=rho*qes**2*thetas

    M=rho*qes*deltastars

    Reth=thetas*qes*rho/mu

    Mes=qes/a

    Hk=(Hs-0.290*Mes**2)/(1.0+0.113*Mes**2)

    Cf=2*(-0.067+0.01977*(7.4-Hk)**2/(Hk-1.0))/Reth
    tau=Cf*qes**2*rho/2

    rvj=residual.get_rv_matrix(xs, ys)
    rdvdxj=residual.get_rdvdx_matrix(xs, ys)
    rdvdyj=residual.get_rdvdy_matrix(xs, ys)
    rudvdxj=residual.get_rudvdx_matrix(xs, ys)
    rudvdyj=residual.get_rudvdy_matrix(xs, ys)

    R=rdvdxj@J+M@rudvdxj@qes-rvj@tau

    rmass, rmomx, rmomz, ren, rts=residual.cell_getresiduals(\
        np.zeros_like(thetas), thetas, Hs, np.zeros_like(thetas), Nts, qes, np.zeros_like(qes), np.zeros_like(qes), \
            rho, a, 50.0, 10.0, \
                mtosys, 1.0, mu, 6.0, 1.4, rvj, rdvdxj, rdvdyj, rudvdxj, rudvdyj)

    assert _arr_compare(rmomx, R, tol=1e-2, relative=rmomx)

    Hstar=1.515+0.076*(4.0-Hk)**2/Hk

    CD=(0.001025*(4.0-Hk)**5.5+0.1035)*Hstar/Reth

    D=rho*qes**3*CD

    Hprime=(0.064/(Hk-0.8)+0.251)*Mes**2

    deltaprime=Hprime*thetas

    rhoQ=deltaprime*rho*qes

    thetastar=Hstar*thetas

    E=thetastar*qes**3*rho

    R=rdvdxj@E+rhoQ@rudvdxj@qes**2-2*rvj@D

    assert _arr_compare(ren, R, tol=1e-2, relative=R)

    p=pfunc(Reth, Hk, thetas, 10.0)

    R=qes@rudvdxj@Nts-rvj@(p*qes)

    assert _arr_compare(rts, R, tol=1e-2, relative=R)
