import numpy as np
import numpy.random as rnd
import scipy.sparse as sps
import math

from chochoBL import *
from three_equation_test import std_mesh_fulldata
from xtr_test import _gen_flatplate

import pytest

rho=defatm.rho
mu=defatm.mu

def _perturbations_from_mesh(msh, factor=1e-7):
    return {
        'qx':np.amax(msh.gr.nodes['q'].value['qx'])*factor*np.ones_like(msh.gr.nodes['q'].value['qx']),
        'qy':np.amax(msh.gr.nodes['q'].value['qy'])*factor*np.ones_like(msh.gr.nodes['q'].value['qy']), 
        'qz':np.amax(msh.gr.nodes['q'].value['qz'])*factor*np.ones_like(msh.gr.nodes['q'].value['qz']),
        'th11':np.amax(msh.gr.nodes['th11'].value['th11'])*factor*np.ones_like(msh.gr.nodes['th11'].value['th11']),
        'H':np.amax(msh.gr.nodes['H'].value['H'])*factor*np.ones_like(msh.gr.nodes['H'].value['H']),
        'N':np.amax(msh.gr.nodes['N'].value['N'])*factor*np.ones_like(msh.gr.nodes['N'].value['N']),
        'beta':np.amax(msh.gr.nodes['beta'].value['beta'])*factor*np.ones_like(msh.gr.nodes['beta'].value['beta']),
        'n':np.amax(msh.gr.nodes['n'].value['n'])*factor*np.ones_like(msh.gr.nodes['n'].value['n'])
    }

def total_residual(value):
    return sum(v@v for v in value.values())/2

def test_findiff_small():
    msh1=std_mesh_fulldata()

    perts=_perturbations_from_mesh(msh1)

    msh2=std_mesh_fulldata(perts)

    value1, grad=msh1.calculate_graph()
    value2, _=msh2.calculate_graph()

    var_an=sum(p@g for p, g in zip(perts.values(), grad.values()))

    var_num=(total_residual(value2)-total_residual(value1))

    assert np.abs((var_an-var_num)/var_an)<1e-3

def test_turbulent_findiff():
    assert_diff()

def assert_diff(prop='th11', factor=1e-7):
    Lx=1.0
    Ly=1.0

    Re_target=6e5

    Uinf=Re_target*mu/(Lx*rho)

    msh1, x0_1, q1, xaux, _, indmat=_gen_flatplate(Uinf=Uinf, echo=False, factor=1.0, Lx=Lx, Ly=Ly, nm=10, nn=2, Ncrit=1.0, A_transition=2.0, adj=True)
    msh2, x0_2, q2, xaux, _, _=_gen_flatplate(Uinf=Uinf, echo=False, factor=1.0, Lx=Lx, Ly=Ly, nm=10, nn=2, Ncrit=1.0, A_transition=2.0, adj=True)

    msh1.opt.q=q1
    msh2.opt.q=q2

    x0_2[prop]+=factor

    msh1.opt.calculate(msh1.opt.pack(x0_1))
    msh2.opt.calculate(msh2.opt.pack(x0_2))

    var_an=msh1.opt.grad@(msh2.opt.x-msh1.opt.x)
    var_num=msh2.opt.fx-msh1.opt.fx

    print(var_an, var_num)

    assert np.abs((var_an-var_num)/var_an)<1e-3
