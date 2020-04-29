import numpy as np
import scipy.optimize as sopt
import time as tm

import fluids.atmosphere as atm

import abaqus as abq
import closure as clsr

class station:
    def __init__(self, clsr, delta=0.0, drhoq_dx=0.0, drhoq_dz=0.0, d2rhoq_dx2=0.0, d2rhoq_dxdz=0.0, beta=0.0, dbeta_dx=0.0, dbeta_dz=0.0, deltastar=0.0, props=atm.ATMOSPHERE_1976(Z=0.0, dT=0.0), \
        qe=1.0, gamma=1.4):
        self.clsr=clsr
        self.delta=delta
        self.deltastar=0.0
        self.Lambda=delta**2*drhoq_dx/props.mu
        self.drhoq_dx=drhoq_dx
        self.drhoq_dz=drhoq_dz
        self.d2rhoq_dx2=d2rhoq_dx2
        self.d2rhoq_dxdz=d2rhoq_dxdz
        self.beta=beta
        self.dbeta_dx=dbeta_dx
        self.dbeta_dz=dbeta_dz
        self.atm_props=props
        self.qe=qe
        self.gamma=gamma
        self.Me=qe/props.v_sonic
        self.dyn_press=gamma*props.P*self.Me**2/2
        self.rho=2*self.dyn_press/(self.qe**2)
        self.Red=2*self.dyn_press*self.delta/(qe*props.mu)
        self.pp_w=clsr.ap_w+clsr.bp_w*self.Lambda #derivative of p at the wall
    def calc_derivs(self, dd_dx_seed=1.0, dd_dz_seed=0.0):
        #tier 1 definitions
        dLambda_dx=(2*dd_dx_seed*self.drhoq_dx+self.delta*self.d2rhoq_dx2)*self.delta/self.atm_props.mu
        dLambda_dz=(2*dd_dz_seed*self.drhoq_dx+self.delta*self.d2rhoq_dxdz)*self.delta/self.atm_props.mu
        
        dRed_dx=(dd_dx_seed*self.rho*self.qe+self.delta*self.drhoq_dx)/self.atm_props.mu
        dRed_dz=(dd_dz_seed*self.rho*self.qe+self.delta*self.drhoq_dz)/self.atm_props.mu

        dUt_dx=-(dRed_dx*self.Ut**2*(1.0-np.cos(self.beta))+self.Red*self.Ut**2*np.sin(self.beta)*self.dbeta_dx+dLambda_dx*self.clsr.bp_w*self.C_Ut_dp\
            -self.pp_w*self.Ut**2*self.up_prime_edge*dRed_dx)/(2*self.Red*self.Ut*(1.0-np.cos(self.beta))-self.pp_w*(self.up_edge+self.Red*self.Ut*self.up_prime_edge))
        dUt_dz=-(dRed_dz*self.Ut**2*(1.0-np.cos(self.beta))+self.Red*self.Ut**2*np.sin(self.beta)*self.dbeta_dz+dLambda_dz*self.clsr.bp_w*self.C_Ut_dp\
            -self.pp_w*self.Ut**2*self.up_prime_edge*dRed_dz)/(2*self.Red*self.Ut*(1.0-np.cos(self.beta))-self.pp_w*(self.up_edge+self.Red*self.Ut*self.up_prime_edge))

        ddeltastar_dx=dRed_dx*self.Ut+self.Red*dUt_dx
        ddeltastar_dz=dRed_dz*self.Ut+self.Red*dUt_dz

        dC_dx=self.dC_dUt*dUt_dx+self.dC_ddp*ddeltastar_dx
        dC_dz=self.dC_dUt*dUt_dz+self.dC_ddp*ddeltastar_dz

        dKsi_W_dx=self.dKsi_W_ddp*ddeltastar_dx
        dKsi_W_dz=self.dKsi_W_ddp*ddeltastar_dz
        dKsi_W2_dx=self.dKsi_W2_ddp*ddeltastar_dx
        dKsi_W2_dz=self.dKsi_W2_ddp*ddeltastar_dz
        dKsi_Wa_dx=self.dKsi_Wa_ddp*ddeltastar_dx
        dKsi_Wa_dz=self.dKsi_Wa_ddp*ddeltastar_dz
        dKsi_Wb_dx=self.dKsi_Wb_ddp*ddeltastar_dx
        dKsi_Wb_dz=self.dKsi_Wb_ddp*ddeltastar_dz
        dKsi_WM_dx=self.dKsi_WM_ddp*ddeltastar_dx
        dKsi_WM_dz=self.dKsi_WM_ddp*ddeltastar_dz
        dKsi_WMa_dx=self.dKsi_WMa_ddp*ddeltastar_dx
        dKsi_WMa_dz=self.dKsi_WMa_ddp*ddeltastar_dz
        dKsi_WMb_dx=self.dKsi_WMb_ddp*ddeltastar_dx
        dKsi_WMb_dz=self.dKsi_WMb_ddp*ddeltastar_dz
        dKsi_W2M_dx=self.dKsi_W2M_ddp*ddeltastar_dx
        dKsi_W2M_dz=self.dKsi_W2M_ddp*ddeltastar_dz
        dKsi_W2M2_dx=self.dKsi_W2M2_ddp*ddeltastar_dx
        dKsi_W2M2_dz=self.dKsi_W2M2_ddp*ddeltastar_dz
    def calc_data(self, Ut_initguess=0.1):
        self.turb_deduce(Ut_initguess=Ut_initguess)
        
        tanb=np.tan(self.beta)
        self.F=self.Ut*self.Ksi_W+self.C_Ut_dp*(self.clsr.Ksi_a+self.Lambda*self.clsr.Ksi_b)
        self.G=tanb*(self.Ut*self.Ksi_WM+self.C_Ut_dp*(self.clsr.Ksi_Ma+self.clsr.Ksi_Mb*self.Lambda))
        self.H=tanb*(self.Ut**2*self.Ksi_W2M+2*self.C_Ut_dp*self.Ut*(self.Ksi_WMa+self.Ksi_WMb*self.Lambda)+\
            self.C_Ut_dp**2*(self.clsr.Ksi_Ma2+2*self.Lambda*self.clsr.Ksi_Mab+self.Lambda**2*self.clsr.Ksi_Mb2))
        self.I=self.Ut**2*self.Ksi_W2+2*self.C_Ut_dp*self.Ut*(self.Ksi_Wa+self.Ksi_Wb*self.Lambda)+\
            self.C_Ut_dp**2*(self.clsr.Ksi_a2+2*self.Lambda*self.clsr.Ksi_ab+self.Lambda**2*self.clsr.Ksi_b2)
        self.J=tanb**2*(self.Ut**2*self.Ksi_W2M2+2*self.C_Ut_dp*self.Ut*(self.Ksi_WM2a+self.Ksi_WM2b*self.Lambda)+\
            self.C_Ut_dp**2*(self.clsr.Ksi_M2a2+2*self.clsr.Ksi_M2ab*self.Lambda+self.clsr.Ksi_M2b2*self.Lambda**2))
        
        self.deltax_bar=1.0-self.F
        self.deltaz_bar=self.G
        self.Thetaxx=self.F-self.I
        self.Thetaxz=self.G-self.H
        self.Thetazx=self.H
        self.Thetazz=self.J
    def turb_deduce(self, Ut_initguess=0.1):
        self.Ut=sopt.fsolve(self.turb_it, x0=Ut_initguess)[0]
        self.deltastar=self.Ut*self.Red
        self.Tau_w=self.Ut**2*2*self.dyn_press
        self.up_edge=self.clsr.LOTW(self.deltastar)
        self.C_Ut_dp=(1.0-self.Ut*self.up_edge)
        h=self.deltastar/self.clsr.Ksi_disc
        self.up_prime_edge=(self.clsr.LOTW(self.deltastar+h)-self.clsr.LOTW(self.deltastar-h))/(2*h)

        #obtain Ksis from closure relationships
        self.Ksi_W=self.clsr.Ksi_W(self.deltastar)
        self.dKsi_W_ddp=self.clsr.Ksi_W(self.deltastar, dx=1)
        self.Ksi_W2=self.clsr.Ksi_W2(self.deltastar)
        self.dKsi_W2_ddp=self.clsr.Ksi_W2(self.deltastar, dx=1)
        self.Ksi_Wa=self.clsr.Ksi_Wa(self.deltastar)
        self.dKsi_Wa_ddp=self.clsr.Ksi_Wa(self.deltastar, dx=1)
        self.Ksi_Wb=self.clsr.Ksi_Wb(self.deltastar)
        self.dKsi_Wb_ddp=self.clsr.Ksi_Wb(self.deltastar, dx=1)
        self.Ksi_WM=self.clsr.Ksi_WM(self.deltastar)
        self.dKsi_WM_ddp=self.clsr.Ksi_WM(self.deltastar, dx=1)
        self.Ksi_WMa=self.clsr.Ksi_WMa(self.deltastar)
        self.dKsi_WMa_ddp=self.clsr.Ksi_WMa(self.deltastar, dx=1)
        self.Ksi_WMb=self.clsr.Ksi_WMb(self.deltastar)
        self.dKsi_WMb_ddp=self.clsr.Ksi_WMb(self.deltastar, dx=1)
        self.Ksi_W2M=self.clsr.Ksi_W2M(self.deltastar)
        self.dKsi_W2M_ddp=self.clsr.Ksi_W2M(self.deltastar, dx=1)
        self.Ksi_W2M2=self.clsr.Ksi_W2M2(self.deltastar)
        self.dKsi_W2M2_ddp=self.clsr.Ksi_W2M2(self.deltastar, dx=1)
        self.Ksi_WM2a=self.clsr.Ksi_WM2a(self.deltastar)
        self.dKsi_WM2a_ddp=self.clsr.Ksi_WM2a(self.deltastar, dx=1)
        self.Ksi_WM2b=self.clsr.Ksi_WM2b(self.deltastar)
        self.dKsi_WM2b_ddp=self.clsr.Ksi_WM2b(self.deltastar, dx=1)

        self.dC_dUt=-self.up_edge
        self.dC_ddp=-self.up_prime_edge*self.Ut
    def turb_it(self, Ut): #function defining a single iteration for Newton's method
        return self.Red*Ut**2*(1.0-np.cos(self.beta))+self.pp_w*(1.0-Ut*self.clsr.LOTW(self.Red*Ut))

defclsr=clsr.closure(deltastar_disc=40)
stat=station(defclsr, delta=1.0, qe=1.0)
t=tm.time()
stat.calc_data()
stat.calc_derivs()
stat.calc_derivs()
stat.calc_derivs()
stat.calc_derivs()
print(tm.time()-t)