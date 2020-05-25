import numpy as np
import scipy.sparse as sps

from chochoBL import *

import pytest

def _arr_compare(a, b, tol=1e-5):
    return np.all(np.abs(a-b)<tol)

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
