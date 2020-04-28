import numpy as np
import scipy.optimize as sopt
import time as tm

import fluids.atmosphere as atm

import abaqus as abq
import closure as clsr

class station:
    def __init__(self, clsr, delta=0.0, drhoq_dx=0.0, drhow_x=0.0, d2rhoq_dx2=0.0, deltastar=0.0, props=atm.ATMOSPHERE_1976(Z=0.0, dT=0.0), \
        qe=1.0):
        self.clsr=clsr
        self.delta=delta
        self.deltastar=0.0
        self.Lambda=delta**2*drhoq_dx/props.mu
        self.beta=np.arctan2(drhow_x, drhoq_dx)
        self.atm_props=props
        self.qe=qe
        self.Red=qe*delta*props.rho/props.mu
        self.pp_w=clsr.ap_w*self.Lambda+clsr.bp_w #derivative of p at the wall
    def calc_derivs(self, dd_dx_seed=1.0, dd_dz_seed=0.0):
        #tier 1 definitions
        pass
    def turb_deduce(self, Ut_initguess=0.05):
        t=tm.time()
        soln=sopt.fsolve(self.turb_it, x0=Ut_initguess)[0]
        print(tm.time()-t)
        return soln
    def turb_it(self, Ut): #function defining a single iteration for Newton's method
        return self.Red*Ut**2*(1.0-np.cos(self.beta))+self.pp_w*(1.0-Ut*self.clsr.LOTW(self.Red*Ut))

defclsr=clsr.closure()
stat=station(defclsr, delta=1.0, qe=1.0)
print(stat.turb_deduce())