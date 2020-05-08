import numpy as np
import scipy.interpolate as sinterp
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from closure import *

class turbulence_abaqus:
    def __init__(self, deltastars, props):
        self.rule=sinterp.UnivariateSpline(deltastars, props, ext=0) #engage extrapolation
        self.deltastar_lims=(deltastars[0], deltastars[1])
    def __call__(self, deltastar, dx=0):
        return self.rule(deltastar, nu=dx)

def Ksi(foo, disc=100):
    return np.trapz(foo(np.linspace(0.0, 1.0, disc)))

class Ksi_abaqus(turbulence_abaqus):
    def __init__(self, deltastars, foo=lambda x: x, disc=100): #foo necessarily takes arguments eta and deltastar
        props=np.zeros(len(deltastars))
        for i in range(len(deltastars)):
            props[i]=Ksi(foo=lambda eta: foo(eta, deltastars[i]), disc=disc)
        super().__init__(deltastars, props)
    def __call__(self, deltastar, dx=0):
        return super().__call__(deltastar, dx=dx)