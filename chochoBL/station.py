import numpy as np
import numpy.linalg as lg
import math as mth
import scipy.optimize as sopt
import time as tm
import random as rnd

import fluids.atmosphere as atm

import abaqus as abq
from closure import *
import transition as trans

defatm=atm.ATMOSPHERE_1976(Z=0.0, dT=0.0)

class station:
    def __init__(self, turb_clsr=def_turb_closure, lam_clsr=def_lam_closure, delta=0.0, dq_dx=0.0, dq_dz=0.0, d2q_dx2=0.0, d2q_dxdz=0.0, beta=0.0, dbeta_dx=0.0, dbeta_dz=0.0, props=defatm, \
        qe=1.0, Uinf=1.0, gamma=1.4, transition=False, transition_envelope=trans.Tollmien_Schlichting_Drela):
        #closure relationships as input
        self.turb_clsr=turb_clsr
        self.lam_clsr=lam_clsr
        self.transition=transition
        self.transition_envelope=transition_envelope

        #define thicknesses
        self.delta=delta
        self.deltastar=0.0

        #use small disturbance assumptions to calculate local Mach number
        self.Me=qe/props.v_sonic

        #calculate local densities and momentum variation rates
        self.rho=props.rho*(1.0-self.Me**2*(qe-Uinf)/Uinf) #also a small disturbance assumption: Drho/rho=-Me**2*DV/V
        self.drhoq_dx=(1.0-self.Me**2)*self.rho*dq_dx
        self.drhoq_dz=(1.0-self.Me**2)*self.rho*dq_dz
        drho_dx=-self.Me*self.rho*dq_dx/props.v_sonic
        drho_dz=-self.Me*self.rho*dq_dz/props.v_sonic
        self.d2rhoq_dx2=-2*self.Me*(dq_dx**2)*self.rho/props.v_sonic+(1.0-self.Me**2)*(drho_dx*dq_dx+self.rho*d2q_dx2)
        self.d2rhoq_dxdz=-2*self.Me*dq_dx*dq_dz*self.rho/props.v_sonic+(1.0-self.Me**2)*(drho_dz*dq_dx+self.rho*d2q_dxdz)

        #calculating shape parameter Lambda=fpp
        self.Lambda=-delta**2*self.drhoq_dx/props.mu

        #setting crossflow and its variations along the axes
        self.beta=beta
        self.dbeta_dx=dbeta_dx
        self.dbeta_dz=dbeta_dz

        #setting atmospheric properties
        self.atm_props=props

        #setting freestream properties
        self.qe=qe
        self.dq_dx=dq_dx
        self.dq_dz=dq_dz
        self.dyn_press=self.rho*qe**2/2

        #setting local Reynolds number in respect to thickness
        self.Red=self.rho*self.qe*self.delta/props.mu
    
    def _calc_parameter_derivatives(self, dd_dx, dd_dz):
        dLambda_dx=-self.delta*(2*dd_dx*self.drhoq_dx+self.delta*self.d2rhoq_dx2)/self.atm_props.mu
        dLambda_dz=-self.delta*(2*dd_dz*self.drhoq_dx+self.delta*self.d2rhoq_dxdz)/self.atm_props.mu

        dRed_dx=(self.qe*self.rho*dd_dx+self.drhoq_dx*self.delta)/self.atm_props.mu
        dRed_dz=(self.qe*self.rho*dd_dz+self.drhoq_dz*self.delta)/self.atm_props.mu

        return dLambda_dx, dLambda_dz, dRed_dx, dRed_dz

    def _eqns_solve(self, tolerance=1e-5):
        if self.delta!=0.0:
            #calculating local thicknesses according to closure relationships
            #calculating thickness tensor derivatives
            if self.transition:
                th, dx, dz, Cf=self.turb_clsr(self.Lambda, self.Red)
                dth_dLambda, dth_dRed=self.turb_clsr(self.Lambda, self.Red, nu=True)
            else:
                th, dx, dz, Cf=self.lam_clsr(self.Lambda, self.Red)
                dth_dLambda, dth_dRed=self.lam_clsr(self.Lambda, self.Red, nu=True)

            #generating crossflow conversion matrix and its derivatives
            tanb=np.tan(self.beta)
            sec2=np.sqrt(tanb**2+1.0)
            dtanb_dx=self.dbeta_dx*sec2
            dtanb_dz=self.dbeta_dz*sec2
            dtanb2_dx=2*tanb*dtanb_dx
            dtanb2_dz=2*tanb*dtanb_dz

            cross=np.array([[1.0, tanb], [tanb, tanb**2]])
            dcross_dx=np.array([[0.0, dtanb_dx], [dtanb_dx, dtanb2_dx]])
            dcross_dz=np.array([[0.0, dtanb_dz], [dtanb_dz, dtanb2_dz]])

            #now we're due to define linear functions for the derivatives of Lambda and Red in respect to dd_dx and dd_dz
            dl_00x, dl_00z, drd_00x, drd_00z=self._calc_parameter_derivatives(0.0, 0.0)
            dl_11x, dl_11z, drd_11x, drd_11z=self._calc_parameter_derivatives(1.0, 1.0)

            #derivate quantities of interest for dd_dksi=1 and dd_dksi=0.0 (for linearization)
            dth_00x=dth_dLambda*dl_00x+dth_dRed*drd_00x
            dth_11x=dth_dLambda*dl_11x+dth_dRed*drd_11x
            dth_00z=dth_dLambda*dl_00z+dth_dRed*drd_00z
            dth_11z=dth_dLambda*dl_11z+dth_dRed*drd_11z

            #apply crossflow
            dth_00x=dth_00x*cross+th*dcross_dx
            dth_11x=dth_11x*cross+th*dcross_dx
            dth_00z=dth_00z*cross+th*dcross_dz
            dth_11z=dth_11z*cross+th*dcross_dz
            th*=cross
            dz*=tanb

            #saving quantities of interest
            self.th=th*self.delta
            self.dx=dx*self.delta
            self.dz=dz*self.delta
            self.Cf=Cf

            #apply effective thicknesses
            dth_00x=dth_00x*self.delta
            dth_11x=dth_11x*self.delta+th
            dth_00z=dth_00z*self.delta
            dth_11z=dth_11z*self.delta+th

            #solve equations and generate gradient of boundary layer thickness
            RHS=np.zeros(2)
            A=np.zeros((2, 2))

            #complete derivative linearization
            #first quantity of interest: dthxx_dx+dthxz_dz
            #second quantity of interest: dthzx_dx+dthzz_dz
            b=np.array([dth_00x[0, 0]+dth_00z[0, 1], dth_00x[1, 0]+dth_00z[1, 1]])
            A[:, 0]=dth_11x[:, 0]+dth_00z[:, 1]-b
            A[:, 1]=dth_00x[:, 0]+dth_11z[:, 1]-b

            #defining right hand side
            RHS[0]=Cf*np.cos(self.beta)/2-self.dx*self.dq_dx/self.qe-self.dz*self.dq_dz/self.qe-(2.0-self.Me**2)*(self.th[0, 0]*self.dq_dx+self.th[0, 1]*self.dq_dz)/self.qe
            RHS[1]=Cf*np.sin(self.beta)/2+self.delta*self.dq_dx*tanb/self.qe-(2.0-self.Me**2)*(self.th[1, 0]*self.dq_dx+self.th[1, 1]*self.dq_dz)/self.qe

            #solve linear system
            if lg.norm(A[1, :])<tolerance:
                return np.array([(RHS-b)[0]/A[0, 0], 0.0])
            else:
                return lg.solve(A, b=RHS-b)
        else:
            return np.array([0.0, 0.0])

    def calc_data(self):
        '''
        function to calculate local flow data regardless of derivatives
        '''

        #calculating local thicknesses according to closure relationships
        if self.delta!=0.0:
            if self.transition:
                th, dx, dz, Cf=self.turb_clsr(self.Lambda, self.Red)
            else:
                th, dx, dz, Cf=self.lam_clsr(self.Lambda, self.Red)
        else:
            th=np.zeros((2, 2))
            dx=0.0
            dz=0.0
            Cf=0.0

        #applying crossflow
        tanb=np.tan(self.beta)
        cross=np.array([[1.0, tanb], [tanb, tanb**2]])

        #compute thicknesses
        self.th=th*cross*self.delta
        self.dx=dx*self.delta
        self.dz=dz*self.delta
        self.Cf=Cf
    
    def has_transition(self):
        return self.transition_envelope(self) or self.transition

    def calcpropag(self, tolerance=1e-5):
        return self._eqns_solve(tolerance=tolerance)