import numpy as np
import time as tm

import pytest

from chochoBL import *

def _test_mesh():
    pts=np.zeros((3, 3, 3))

    pts[0, :, 0]=0.0
    pts[1, :, 0]=1.0
    pts[2, :, 0]=2.0
    pts[:, 0, 1]=0.0
    pts[:, 1, 1]=1.0
    pts[:, 2, 1]=2.0

    msh=mesh()

    ndinds=np.zeros((3, 3), dtype='int')

    for i in range(3):
        for j in range(3):
            ndinds[i, j]=msh.add_node(pts[i, j, :])

    for i in range(2):
        for j in range(2):
            msh.add_cell({ndinds[i, j], ndinds[i, j+1], ndinds[i+1, j+1], ndinds[i+1, j]})

    normals=np.zeros((9, 3))
    normals[:, 0]=0.0
    normals[:, 1]=0.0
    normals[:, 2]=1.0

    msh.compose(normals)

    return msh

def test_meshgen():
    '''
    Test with 2x2 cell mesh
    '''

    msh=_test_mesh()

    ideal_sets=[
        {0, 1, 3, 4},
        {1, 2, 5, 4},
        {3, 4, 6, 7},
        {4, 5, 8, 7}
    ]

    ideal_nodes=[
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 2.0, 0.0]
    ]

    ideal_Mtosys=np.array(
        [
            [1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0]
        ]
    )

    assert all([np.all(c.Mtosys==ideal_Mtosys) for c in msh.cells]), "Mesh failed in coordinate system generation"

def test_v_resint():
    '''
    Test integration of function over an area
    '''

    msh=_test_mesh()

    xs=msh.nodes[:, 0]

    a=2.0 # angular coefficient
    b=1.0 # linear coefficient

    f=a*xs+b

    distJ=msh.dcell_dnode_compose((f,))

    f_c=distJ@f

    R=np.sum(msh.v_res_Jac@f_c)

    Lx=2.0
    Ly=2.0

    R_an=Ly*(a*Lx**2/2+b*Lx)

    assert np.abs(R_an-R)<1e-3*np.abs(R_an)

def test_dvdx_resint():
    '''
    Test integration of function over an area
    '''

    msh=_test_mesh()

    xs=msh.nodes[:, 0]

    a=2.0 # angular coefficient
    b=1.0 # linear coefficient

    f=a*xs+b

    distJ=msh.dcell_dnode_compose((f,))

    f_c=distJ@f

    R=0.0
    for i, c in enumerate(msh.cells):
        R+=np.sum(c.Rdvdx@f_c[4*i:4*(i+1)])

    Lx=2.0
    Ly=2.0

    R_an=Ly*(a*Lx)

    assert np.abs(R_an-R)<=1e-3*np.abs(R_an)
