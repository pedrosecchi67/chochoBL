import numpy as np
import scipy.sparse as sps

from chochoBL import *

import pytest

def test_q2_conversion():
    q2f=q2_fset(2)

    passive={
        'normal':np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0]
            ]
        )
    }

    qp1=np.array([1.0, 0.0, -1.0])
    qp2=np.array([3.0, 0.0, 1.0])

    q1x=np.array([qp1[0], qp2[0]])
    q1y=np.array([qp1[1], qp2[1]])
    q1z=np.array([qp1[2], qp2[2]])

    arglist=[q1x, q1y, q1z]

    output=q2f(arglist, passive)

    J=q2f.Jacobian(arglist, passive=passive, mtype='lil')

    outlist=q2f.out_unpack(output)

    assert np.all(outlist[0]==np.array([1.0, -1.0])), "q2x calculation failed"
    assert np.all(outlist[1]==np.array([0.0, 0.0])), "q2y calculation failed"
    assert np.all(outlist[2]==np.array([1.0, 3.0])), "q2z calculation failed"
    
    assert np.all(output==J.todense()@np.hstack((q1x, q1y, q1z))), "Jacobian calculation failed"
