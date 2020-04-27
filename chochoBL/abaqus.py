import numpy as np
import scipy.interpolate as sinterp
import closure as clsr

class turbulence_abaqus:
    def __init__(self, deltastars, props):
        self.rule=sinterp.UnivariateSpline(deltastars, props, ext=0) #engage extrapolation
        self.deltastar_lims=(deltastars[0], deltastars[1])
    def __call__(self, deltastar):
        return self.rule(deltastar)

def Ksi(foo, disc=100):
    return np.trapz(foo(np.linspace(0.0, 1.0, disc)))

def Ksi_abaqus(turbulence_abaqus):
    def __init__(self, deltastars, foos=(), foo_degs=(), LOTW=clsr.Gersten_Herwig_LOTW, LOTW_deg=0):
        super().__init__(self)