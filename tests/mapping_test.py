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

def findiff(map, val, factor=1e-7):
    v1, j=map(val)
    v2, _=map(val+val*factor)

    var_an=(val*factor)*j
    var_num=(v2-v1)

    print(var_an, var_num)

    if type(val)==np.ndarray:
        assert np.all(np.abs(var_an-var_num)<=np.abs(var_an)*findiff_reltol)
    else:
        assert np.abs(var_an-var_num)<=np.abs(var_an)*findiff_reltol
