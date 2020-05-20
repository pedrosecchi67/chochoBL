import numpy as np
import scipy.sparse as sps

import pytest

from chochoBL import *

def test_function_evaluation():
    '''
    Test for function class creation and derivative evaluation
    '''

    arglist=[np.array([0.0, 1.0]), np.array([1.0, 0.0])]

    f=func(f=lambda x: x@x, derivs=(lambda x: 2*x,), args=[0])
    g=func(f=lambda x: x, derivs=(lambda x: np.eye(len(x)),), args=[1])
    h=func(f=lambda x, y: x*y, derivs=(lambda x, y: np.diag(y), lambda x, y: np.diag(x)), args=[0, 1])

    assert f(arglist)==1.0, "f evaluation failed"
    assert np.all(g(arglist)==np.array([1.0, 0.0])), "g evaluation failed"

    assert np.all(f.Jacobian(arglist)==np.array([0.0, 2.0])), "f Jacobian evaluation failed"
    assert np.all(g.Jacobian(arglist)==np.eye(2)), "g Jacobian evaluation failed"

    fset=funcset(fs=[f, g, h], arglens=[2, 2], outlens=[1, 2, 2])

    assert np.all(fset(arglist)==np.array([1.0, 1.0, 0.0, 0.0, 0.0])), "Function set evaluation failed"
    assert np.all(
        fset.Jacobian(arglist)==np.array(
            [
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        )
    ), "Function set Jacobian evaluation failed"

def test_vector_mixing():
    a=np.arange(0, 3, 1, dtype='int')
    b=np.arange(3, 6, 1, dtype='int')

    v, J=mix_vectors((a, b))

    assert np.all(v==np.array([0, 3, 1, 4, 2, 5])), "Vector mixing failed"

def test_vector_reordering_Jacobian():
    a=np.array([0, 1, 2])
    b=np.array([1, 0, 2])

    J=reorder_Jacobian(b, 3)

    assert np.all(J@a==np.array([1, 0, 2])), "Vector reordering Jacobian failed"

def test_correspondence_Jacobian():
    a=np.arange(0.0, 12.0, 2.0)
    b=np.arange(1.0, 13.0, 2.0)

    correspondence=np.array(
        [
            [0, 1, 4, 3],
            [1, 2, 5, 4]
        ], dtype='int'
    )

    J=dcell_dnode_Jacobian((a, b), correspondence)

    ideal_cell_mix=np.array(
        [0.0, 2.0, 8.0, 6.0, 1.0, 3.0, 9.0, 7.0, 2.0, 4.0, 10.0, 8.0, 3.0, 5.0, 11.0, 9.0]
    )

    assert np.all(J@np.hstack((a, b))==ideal_cell_mix), "Node to cell Jacobian test failed"

def test_LT_node_mix():
    T=np.array(
        [
            [1.0, 2.0, 0.0],
            [1.0, 0.0, -9.0],
            [5.0, -1.0, 0.0]
        ]
    )

    a=np.array([0.0, 1.0, -1.0])
    b=np.array([1.0, 0.0, -1.0])
    c=np.array([4.0, 1.0, -4.0])
    d=np.array([1.0, 0.0, 0.0])

    Ta=T@a
    Tb=T@b
    Tc=T@c
    Td=T@d

    Tnew=LT_node_mix(T)
    mixvars, _=mix_vectors((a, b, c, d))
    mixres, _=mix_vectors((Ta, Tb, Tc, Td))

    assert np.all(Tnew@mixvars==mixres), "LT node mixing test failed"
