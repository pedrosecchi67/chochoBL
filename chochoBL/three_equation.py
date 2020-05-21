import numpy as np
import scipy.sparse as sps

'''
Module containing functions and classes referrant to the application of a three-equation
boundary layer solver
'''

from differentiation import *

def _linewise_Hadamard(x, B):

    try:
        return ((B.T).multiply(x)).T
    except:
        return np.multiply(x, B.T).T

def q2x_fromq1(q1y, q1z, passive):

    return passive['normal'][:, 2]*q1y-passive['normal'][:, 1]*q1z

def q2x_dq1y(q1y, q1z, passive):

    return _linewise_Hadamard(passive['normal'][:, 2], sps.eye(len(q1y), format='lil'))

def q2x_dq1z(q1y, q1z, passive):

    return -_linewise_Hadamard(passive['normal'][:, 1], sps.eye(len(q1z), format='lil'))

def q2y_fromq1(q1x, q1z, passive):

    return passive['normal'][:, 0]*q1z-passive['normal'][:, 2]*q1x

def q2y_dq1x(q1x, q1z, passive):

    return -_linewise_Hadamard(passive['normal'][:, 2], sps.eye(len(q1x), format='lil'))

def q2y_dq1z(q1x, q1z, passive):

    return _linewise_Hadamard(passive['normal'][:, 0], sps.eye(len(q1z), format='lil'))

def q2z_fromq1(q1x, q1y, passive):

    return passive['normal'][:, 1]*q1x-passive['normal'][:, 0]*q1y

def q2z_dq1x(q1x, q1y, passive):

    return _linewise_Hadamard(passive['normal'][:, 1], sps.eye(len(q1x), format='lil'))

def q2z_dq1y(q1x, q1y, passive):

    return -_linewise_Hadamard(passive['normal'][:, 0], sps.eye(len(q1y), format='lil'))

def q2_fset(npan):
    '''
    Function returning a function set taking args q1x, q1y, q1z and returning q2x, q2y, q2z 
    (see notation in Drela's IBL3 presentation)
    '''

    fx=func(q2x_fromq1, (q2x_dq1y, q2x_dq1z,), [1, 2], haspassive=True, sparse=True)
    fy=func(q2y_fromq1, (q2y_dq1x, q2y_dq1z,), [0, 2], haspassive=True, sparse=True)
    fz=func(q2z_fromq1, (q2z_dq1x, q2z_dq1y,), [0, 1], haspassive=True, sparse=True)

    return funcset(fs=[fx, fy, fz], arglens=[npan, npan, npan], outlens=[npan, npan, npan], sparse=True)