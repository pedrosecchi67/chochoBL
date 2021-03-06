import numpy as np
import numpy.random as rnd
import scipy.sparse as sps
import math

from chochoBL import *

import pytest

def _arr_compare(a, b, tol=1e-5, min_div=1e-14, relative=None):
    if relative is None:
        return np.all(np.abs(a-b)<tol)
    else:
        return np.all(np.abs(a-b)/\
            np.array([max([min_div, np.abs(r)]) for r in relative])<=tol)

def _get_test_mesh():
    msh=mesh(Uinf=3.0)

    node_coords=[
        [0.0, 0.0, 0.0], #0
        [1.0, 0.0, 0.0], #1
        [2.0, 0.0, 0.0], #2
        [0.0, 1.0, 0.0], #3
        [1.0, 1.0, 0.0], #4
        [2.0, 1.0, 0.0], #5
        [0.0, 2.0, 0.0], #6
        [1.0, 2.0, 0.0], #7
        [2.0, 2.0, 0.0] #8
    ]

    node_indexes=[
        {0, 1, 4, 3},
        {1, 2, 5, 4},
        {3, 4, 7, 6},
        {4, 5, 8, 7}
    ]

    normals=np.zeros((9, 3))
    normals[:, 2]=1.0

    for p in node_coords:
        msh.add_node(p)
    
    for indset in node_indexes:
        msh.add_cell(indset)
    
    msh.compose(normals)

    return msh

def std_mesh_fulldata(perturbations={}):
    msh=_get_test_mesh()

    vels=np.array(
        [
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, -1.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0]
        ]
    )

    H=np.linspace(3.5, 2.2, 9)
    th11=10.0**np.linspace(-4.0, -1.0, 9)
    N=np.linspace(0.0, 10.0, 9)
    beta=np.linspace(-math.radians(45.0), math.radians(45.0), 9)
    n=np.zeros_like(N)

    msh.graph_init()

    data={'q':{'qx':vels[:, 0], 'qy':vels[:, 1], 'qz':vels[:, 2]}, 'th11':{'th11':th11}, 'H':{'H':H}, 'N':{'N':N}, 'beta':{'beta':beta}, 'n':{'n':n}}

    for n in data:
        for d in data[n]:
            if d in perturbations:
                data[n][d]+=perturbations[d]

    for n in data:
        msh.gr.heads[n].set_value(data[n])

    return msh

def _perturbations_from_mesh(msh, factor=1e-7):
    return {
        'qx':np.amax(msh.gr.nodes['q'].value['qx'])*factor,
        'qy':np.amax(msh.gr.nodes['q'].value['qy'])*factor, 
        'qz':np.amax(msh.gr.nodes['q'].value['qz'])*factor,
        'th11':np.amax(msh.gr.nodes['th11'].value['th11'])*factor,
        'H':np.amax(msh.gr.nodes['H'].value['H'])*factor,
        'N':np.amax(msh.gr.nodes['N'].value['N'])*factor,
        'beta':np.amax(msh.gr.nodes['beta'].value['beta'])*factor,
        'n':np.amax(msh.gr.nodes['n'].value['n'])*factor
    }

def _findiff_testprops(props=[], ends=[], tol=1e-3, min_div=1e-14):
    msh1=std_mesh_fulldata()

    pert=_perturbations_from_mesh(msh1)

    msh2=std_mesh_fulldata(pert)

    msh1.gr.calculate(ends)
    msh2.gr.calculate(ends)

    for p in props:
        val1, n=msh1.gr.get_value(p)
        val2, _=msh2.gr.get_value(p)

        argnames=n.args_to_inds

        args1=[msh1.gr.get_value(argname)[0] for argname in argnames]
        args2=[msh2.gr.get_value(argname)[0] for argname in argnames]

        var_an=np.zeros_like(val1)
        var_num=val2-val1

        for argname, (arg1, arg2) in zip(argnames, zip(args1, args2)):
            J=n.Jac[p][argname]

            if not J is None:
                if len(J.shape)==2:
                    var_an+=J@(arg2-arg1)
                else:
                    var_an+=prod_as_diag(J, arg2-arg1)
        
        #print(var_an, var_num)
        
        assert _arr_compare(var_an, var_num, tol=tol, relative=var_an, min_div=min_div), "Property %s calculation failed" % (p,)

def test_uw_conversion():
    msh=_get_test_mesh()

    vels=np.array(
        [
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, -1.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0]
        ]
    )

    msh.graph_init()

    msh.gr.heads['q'].set_value({'qx':vels[:, 0], 'qy':vels[:, 1], 'qz':vels[:, 2]})

    msh.gr.nodes['uw'].calculate()

    ideal_ulists=[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 3.0, 3.0], [2.0, 2.0, 3.0, 3.0]]
    ideal_wlists=[[1.0, 0.0, 0.0, 1.0], [0.0, -1.0, -1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 0.0, -1.0]]

    def compare_lists(a, b):
        return all([sorted(l1)==sorted(l2) for l1, l2 in zip(a, b)])
    
    us=msh.gr.nodes['uw'].value['u']
    ws=msh.gr.nodes['uw'].value['w']

    real_ulists=[list(us[4*i:4*(i+1)]) for i in range(4)]
    real_wlists=[list(ws[4*i:4*(i+1)]) for i in range(4)]

    assert compare_lists(ideal_ulists, real_ulists), "u velocity calculation failed"
    assert compare_lists(ideal_wlists, real_wlists), "w velocity calculation failed"

    #there's no need to test the Jacobian, since the functions involved are simple LTs and matrix multiplications

def test_qe():
    msh=_get_test_mesh()

    vels=np.array(
        [
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, -1.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0]
        ]
    )

    msh.graph_init()

    msh.gr.heads['q'].set_value({'qx':vels[:, 0], 'qy':vels[:, 1], 'qz':vels[:, 2]})

    msh.gr.nodes['qe'].calculate()
    msh.gr.nodes['uw'].calculate()

    qes=msh.gr.nodes['qe'].value['qe']

    dq_dx=msh.gr.nodes['qe'].Jac['qe']['qx']
    dq_dy=msh.gr.nodes['qe'].Jac['qe']['qy']
    dq_dz=msh.gr.nodes['qe'].Jac['qe']['qz']

    assert _arr_compare(qes, np.sqrt(vels[:, 0]**2+vels[:, 1]**2+vels[:, 2]**2)), "External velocity evaluation failed"

    assert _arr_compare(dq_dx.todense(), np.diag(vels[:, 0]/qes)) and \
        _arr_compare(dq_dy.todense(), np.diag(vels[:, 1]/qes)) and \
            _arr_compare(dq_dz.todense(), np.diag(vels[:, 2]/qes)), \
                "External velocity derivative evaluation failed"

def test_Me_rho():
    msh=_get_test_mesh()

    vels=np.array(
        [
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, -1.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0]
        ]
    )

    msh.graph_init()

    msh.gr.heads['q'].set_value({'qx':vels[:, 0], 'qy':vels[:, 1], 'qz':vels[:, 2]})

    msh.gr.nodes['rho'].calculate()

    qes=np.sqrt(vels[:, 0]**2+vels[:, 1]**2+vels[:, 2]**2)

    a=msh.passive['atm'].v_sonic
    Uinf=msh.passive['Uinf']
    rho0=msh.passive['atm'].rho

    rho_ideal=(1.0-(qes/a)**2*(qes-Uinf)/Uinf)*rho0
    drho_dM=np.diag((-(qes/a)**2*(a-Uinf)/Uinf-2*(qes/a)*(qes-Uinf)/Uinf)*rho0)
    
    assert _arr_compare(msh.gr.nodes['Me'].value['Me'], qes/a), "External Mach number computation failed"
    assert _arr_compare(msh.gr.nodes['Me'].Jac['Me']['qe']@qes, msh.gr.nodes['Me'].value['Me']), "External Mach number Jacobian computation failed"

    assert _arr_compare(msh.gr.nodes['rho'].value['rho'], rho_ideal), "External air density computation failed"
    assert _arr_compare(msh.gr.nodes['rho'].Jac['rho']['Me'].todense(), drho_dM), "External air density Jacobian computation failed"

def test_Reth():
    msh=_get_test_mesh()

    vels=np.array(
        [
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, -1.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0]
        ]
    )

    th11=np.linspace(2.5, 3.5, 9)

    msh.graph_init()

    msh.gr.heads['q'].set_value({'qx':vels[:, 0], 'qy':vels[:, 1], 'qz':vels[:, 2]})
    msh.gr.heads['th11'].set_value({'th11':th11})

    msh.gr.nodes['Reth'].calculate()

    mu=msh.passive['atm'].mu
    rho=msh.gr.nodes['rho'].value['rho']
    qe=msh.gr.nodes['qe'].value['qe']

    Reth=msh.gr.nodes['Reth'].value['Reth']

    assert _arr_compare(msh.gr.nodes['Reth'].value['Reth'], qe*rho*th11/mu), "Momentum thickness Reynolds number calculation failed"

    assert _arr_compare(msh.gr.nodes['Reth'].Jac['Reth']['qe'], np.diag(Reth/qe)) and \
        _arr_compare(msh.gr.nodes['Reth'].Jac['Reth']['rho'], np.diag(Reth/rho)) and \
            _arr_compare(msh.gr.nodes['Reth'].Jac['Reth']['rho'], np.diag(Reth/rho)), "Momentum thickness Reynolds number Jacobian calculation failed"

def test_Hk():
    msh=_get_test_mesh()

    vels=np.array(
        [
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, -1.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0]
        ]
    )

    H=np.linspace(2.5, 3.5, 9)

    msh.graph_init()

    msh.gr.heads['q'].set_value({'qx':vels[:, 0], 'qy':vels[:, 1], 'qz':vels[:, 2]})
    msh.gr.heads['H'].set_value({'H':H})

    msh.gr.nodes['Hk'].calculate()

    Me=msh.gr.nodes['Me'].value['Me']

    Hk_expected=Hk(H, Me)

    assert _arr_compare(Hk_expected, msh.gr.nodes['Hk'].value['Hk']), \
        "Hk value computation failed"

    dHk_dH_expected=dHk_dH(H, Me)
    dHk_dH_real=msh.gr.nodes['Hk'].Jac['Hk']['H']

    assert _arr_compare(dHk_dH_expected, dHk_dH_real), "Hk H Jacobian evaluation failed"

    dHk_dMe_expected=dHk_dMe(H, Me)
    dHk_dMe_real=msh.gr.nodes['Hk'].Jac['Hk']['Me']

    assert _arr_compare(dHk_dMe_expected, dHk_dMe_real), "Hk Me Jacobian evaluation failed"

def test_Hk_findiff():
    _findiff_testprops(props=['Hk'], ends=['closure', 'p', 'uw', 'A'])

def test_Hstar():
    _findiff_testprops(props=['Hstar'], ends=['closure', 'p', 'uw', 'A'])

def test_Hprime():
    _findiff_testprops(props=['Hprime'], ends=['closure', 'p', 'uw', 'A'])

def test_Cf():
    _findiff_testprops(props=['Cf'], ends=['closure', 'p', 'uw', 'A'])

def test_Cd():
    _findiff_testprops(props=['Cd'], ends=['closure', 'p', 'uw', 'A'])

def test_p():
    _findiff_testprops(props=['p'], ends=['closure', 'p', 'uw', 'A'])

def test_A():
    _findiff_testprops(props=['A'], ends=['closure', 'p', 'uw', 'A'])

def test_tanb():
    _findiff_testprops(props=['tanb'], ends=['closure', 'p', 'uw', 'A'])

def test_deltastar():
    _findiff_testprops(props=['deltastar_1', 'deltastar_2'], ends=['closure', 'p', 'uw', 'deltastar'])

def test_Cf_2():
    _findiff_testprops(props=['Cf_2'], ends=['closure', 'p', 'uw', 'deltastar', 'Cf'])

def test_theta():
    _findiff_testprops(props=['th12', 'th21', 'th22'], ends=['closure', 'p', 'uw', 'theta', 'Cf'], min_div=1e-12)

def test_thetastar():
    _findiff_testprops(props=['thetastar_1', 'thetastar_2'], ends=['closure', 'p', 'uw', 'thetastar', 'Cf'])

def test_deltaprime():
    _findiff_testprops(props=['deltaprime_1', 'deltaprime_2'], ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cf'])

def test_Cd_2():
    _findiff_testprops(props=['Cd_2'], ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cd'], tol=1e-2)

def test_J():
    _findiff_testprops(props=['Jxx', 'Jxz', 'Jzx', 'Jzz'], ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cd', 'J'], tol=1e-3)

def test_M():
    _findiff_testprops(props=['Mx', 'Mz'], ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cd', 'J', 'M'], tol=1e-3, min_div=1e-10)

def test_E():
    _findiff_testprops(props=['Ex', 'Ez'], ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cd', 'J', 'M', 'E'], tol=1e-3, min_div=1e-10)

def test_rhoQ():
    _findiff_testprops(props=['rhoQx', 'rhoQz'], ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cd', 'J', 'M', 'rhoQ'], tol=1e-3, min_div=1e-10)

def test_D():
    _findiff_testprops(props=['D'], ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cd', 'J', 'M', 'rhoQ', 'D'], tol=1e-3, min_div=1e-10)

def test_tau():
    _findiff_testprops(props=['tau_x', 'tau_z'], ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cd', 'J', 'M', 'rhoQ', 'D', 'tau'], tol=1e-3, min_div=1e-10)

def test_Rmass():
    _findiff_testprops(props=['Rmass'], ends=['Rmass'], tol=1e-3)

def test_RTS():
    _findiff_testprops(props=['RTS'], ends=['RTS'], tol=1e-3)

def test_Rmomx():
    _findiff_testprops(props=['Rmomx'], ends=['Rmomx'], tol=1e-3)

def test_Rmomz():
    _findiff_testprops(props=['Rmomz'], ends=['Rmomz'], tol=1e-3)

def test_Ren():
    _findiff_testprops(props=['Ren'], ends=['Ren'], tol=1e-3)

def test_sigma_N():
    msh=_get_test_mesh()

    N=rnd.random(9)*9.0

    msh.graph_init()

    msh.gr.heads['N'].set_value({'N':N})

    msh.gr.nodes['sigma_N'].calculate()

    SN_expected=sigma_N(N, msh.passive)

    SN_read=msh.gr.nodes['sigma_N'].value['sigma_N']

    dS_dN_expected=np.diag(dsigma_N_dN(N, SN_expected, msh.passive).todense())
    dS_dN_read=np.diag(msh.gr.nodes['sigma_N'].Jac['sigma_N']['N'].todense())

    assert _arr_compare(SN_read, SN_expected), "sigma_N calculation failed"
    assert _arr_compare(dS_dN_read, dS_dN_expected), "sigma_N Jacobian calculation failed"

def test_quad_mesh():
    posits=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
    )

    normals=np.zeros_like(posits)
    normals[:, 2]=1.0

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

    betas=np.zeros_like(thetas)

    deltastars=Hs*thetas

    msh=mesh()

    for i in range(4):
        msh.add_node(posits[i, :])

    msh.add_cell([0, 1, 2, 3])

    msh.compose(normals)
    msh.graph_init()

    msh.set_values({'q':{'qx':qes, 'qy':np.zeros_like(qes), 'qz':np.zeros_like(qes)}, \
        'n':{'n':np.zeros_like(thetas)}, 'th11':{'th11':thetas}, 'N':{'N':np.zeros_like(thetas)}, \
            'H':{'H':Hs}, 'beta':{'beta':betas}})

    value, grad=msh.calculate_graph()

    J=rho*qes**2*thetas

    M=rho*qes*deltastars

    Reth=thetas*qes*rho/mu

    Mes=qes/a

    Hk=(Hs-0.290*Mes**2)/(1.0+0.113*Mes**2)

    Cf=2*(-0.067+0.01977*(7.4-Hk)**2/(Hk-1.0))/Reth
    tau=Cf*qes**2*rho/2

    c=msh.cells[0]

    org=c.indset

    R=c.Rdvdx@J[org]+M[org]@c.Rudvdx@qes[org]-c.Rv@tau[org]

    assert _arr_compare(R, msh.gr.get_value('Rmomx')[0], tol=1e-2, relative=R)

    Hstar=1.515+0.076*(4.0-Hk)**2/Hk

    CD=(0.001025*(4.0-Hk)**5.5+0.1035)*Hstar/Reth

    D=rho*qes**3*CD

    Hprime=(0.064/(Hk-0.8)+0.251)*Mes**2

    deltaprime=Hprime*thetas

    rhoQ=deltaprime*rho*qes

    thetastar=Hstar*thetas

    E=thetastar*qes**3*rho

    R=c.Rdvdx@E[org]+rhoQ[org]@c.Rudvdx@qes[org]**2-2*c.Rv@D[org]

    assert _arr_compare(R, msh.gr.get_value('Ren')[0], tol=1e-2, relative=R)
