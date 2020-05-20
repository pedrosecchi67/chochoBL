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
