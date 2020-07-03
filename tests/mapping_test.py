import numpy.random as rnd
import numpy as np

from chochoBL import *

import pytest

'''
Script containing tests for conform transformations
'''

findiff_reltol=1e-3

def test_smap():
    map=sigma_mapping((2.0, 5.0))

    findiff(map, 2.1)
    findiff(map, 2.0)
    findiff(map, 5.0)
    findiff(map, 4.9)
    findiff(map, 0.0)

    findiff(map, np.array([2.1, 2.0, 5.0, 4.9, 0.0]))

def test_ident():
    map=identity_mapping

    findiff(map, 2.1)
    findiff(map, 2.0)
    findiff(map, 5.0)
    findiff(map, 4.9)
    findiff(map, 0.0)

    findiff(map, np.array([2.1, 2.0, 5.0, 4.9, 0.0]))

    assert map(0.5)[0]==0.5
    assert np.all(map(np.array([0.0, 1.0]))[0]==np.array([0.0, 1.0]))

def test_ksmap():
    map=KS_mapping(2.0, A=0.1)

    findiff(map, 3.0)

def findiff(map, val, factor=1e-7):
    var=np.maximum(np.abs(val)*factor, factor)

    v1, j=map(val)
    v2, _=map(val+var)

    var_an=(var)*j
    var_num=(v2-v1)

    print('=====')
    print(val, j, var_an, var_num)
    print('=====')

    if type(val)==np.ndarray:
        assert np.all(np.abs(var_an-var_num)<=np.abs(var_an)*findiff_reltol) or np.all(np.abs(var_an-var_num)<factor**2)
    else:
        assert np.abs(var_an-var_num)<=np.abs(var_an)*findiff_reltol or np.abs(var_an-var_num)<factor**2
