import numpy as np
import time as tm

import pytest

from chochoBL import *

def test_meshgen():
    '''
    Test with 2x2 cell mesh
    '''
    
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

    assert msh.cells==ideal_sets, "Mesh generation cell indexing failed"
    assert msh.nodes==ideal_nodes, "Mesh nodes incorrectly listed"

    normals=np.zeros((9, 3))
    normals[:, 0]=0.0
    normals[:, 1]=0.0
    normals[:, 2]=1.0

    msh.compose(normals)

    ideal_Mtosys=np.array(
        [
            [1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0]
        ]
    )

    assert all([np.all(c.Mtosys==ideal_Mtosys) for c in msh.cells]), "Mesh failed in coordinate system generation"
