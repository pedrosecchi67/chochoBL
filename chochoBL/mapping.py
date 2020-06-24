import numpy as np
import scipy.special as special

import warnings as wn

'''
Module containing functions and classes for mapping flow variables to
expected ranges via conform transformations
'''

class mapping:
    '''
    Class containing info about a conform transformation
    '''

    def __init__(self, fun_jac, inverse, verification=None, inverse_verificaion=None):
        '''
        Initialize a mapping given a function returning both its value and its Jacobian
        (as a 1-D array if diagonal), an inverse function and a verification function containing expected assertions.
        The verification function should return the inputted vector with the necessary alterations
        applied
        '''

        self.fun_jac=fun_jac
        self.inverse=inverse
        self.inverse_verificaion=inverse_verificaion
        self.verification=verification

    def __call__(self, x):
        '''
        Return the mapped value and its Jacobian, and execute verifications.
        '''

        if not self.verification is None:
            x=self.verification(x)

        f, jac=self.fun_jac(x)

        return f, jac

    def inv(self, x):
        '''
        Return the inverse of the transformation at hand
        '''

        if not self.inverse_verificaion is None:
            x=self.inverse_verificaion(x)

        return self.inverse(x)

def _sigma_fun_jac(x, a, b):
    f=special.expit(x)

    return f*(b-a)+a, f*(1.0-f)*(b-a)

def sigma_inv_verification(x, l, u, tol=1e-7):
    var=u-l

    return np.clip(x, l+tol*var, u-tol*var)

class sigma_mapping(mapping):
    '''
    A conform transformation based on a sigma function
    '''

    def __init__(self, bounds):
        '''
        Define a sigma transformation mapping, based on its bounds for output values
        '''

        super().__init__(lambda x: _sigma_fun_jac(x, bounds[0], bounds[1]), lambda x: special.logit((x-bounds[0])/(bounds[1]-bounds[0])), \
            inverse_verificaion=lambda x: sigma_inv_verification(x, bounds[0], bounds[1]))

_identity_funjac=lambda x: (x, np.ones_like(x))
identity_mapping=mapping(fun_jac=_identity_funjac, inverse=lambda x: x)

def _exp_fun_jac(x, A):
    f=np.exp(x)*A

    return f, f

def log_verification(x, tol=1e-18):
    return np.maximum(x, tol)

class exp_mapping(mapping):
    '''
    A conform transformation based on an exponential function
    '''

    def __init__(self, val0):
        '''
        Initialize conform transformation f(x)=val0.exp(x)
        '''

        super().__init__(lambda x: _exp_fun_jac(x, val0), lambda x: np.log(x/val0), inverse_verificaion=lambda x: log_verification(x/val0))
