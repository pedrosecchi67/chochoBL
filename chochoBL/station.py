import numpy as np
import numpy.linalg as lg
import math as mth
import scipy.optimize as sopt
import time as tm
import random as rnd

import fluids.atmosphere as atm

import abaqus as abq
import closure as clsr

def deriv(Ksi_mat, prod_mat, prodinds=(0, 0, 0), Ksi_inds=(0, 0, 0)):
    return prod_mat[prodinds[0], prodinds[1], prodinds[2], 1]*Ksi_mat[Ksi_inds[0], Ksi_inds[1], Ksi_inds[2], 0]+\
        prod_mat[prodinds[0], prodinds[1], prodinds[2], 0]*Ksi_mat[Ksi_inds[0], Ksi_inds[1], Ksi_inds[2], 1]

defclsr=clsr.closure(deltastar_disc=40)
defatm=atm.ATMOSPHERE_1976(Z=0.0, dT=0.0)

class station:
    def __init__(self, clsr=defclsr, delta=0.0, dq_dx=0.0, dq_dz=0.0, d2q_dx2=0.0, d2q_dxdz=0.0, beta=0.0, dbeta_dx=0.0, dbeta_dz=0.0, props=defatm, \
        qe=1.0, Uinf=1.0, gamma=1.4):
        self.clsr=clsr
        self.delta=delta
        self.deltastar=0.0
        self.Me=qe/props.v_sonic #small disturbance assumption: a~=ainf
        self.rho=props.rho*(1.0-self.Me**2*(qe-Uinf)/Uinf) #also a small disturbance assumption: Drho/rho=-Me**2*DV/V
        self.drhoq_dx=(1.0-self.Me**2)*self.rho*dq_dx
        self.drhoq_dz=(1.0-self.Me**2)*self.rho*dq_dz
        self.Lambda=delta**2*self.drhoq_dx/props.mu
        drho_dx=-self.Me*self.rho*dq_dx/props.v_sonic
        drho_dz=-self.Me*self.rho*dq_dz/props.v_sonic
        self.d2rhoq_dx2=-2*self.Me*(dq_dx**2)*self.rho/props.v_sonic+(1.0-self.Me**2)*(drho_dx*dq_dx+self.rho*d2q_dx2)
        self.d2rhoq_dxdz=-2*self.Me*dq_dx*dq_dz*self.rho/props.v_sonic+(1.0-self.Me**2)*(drho_dz*dq_dx+self.rho*d2q_dxdz)
        self.beta=beta
        self.dbeta_dx=dbeta_dx
        self.dbeta_dz=dbeta_dz
        self.dq_dx=dq_dx
        self.dq_dz=dq_dz
        self.atm_props=props
        self.qe=qe
        self.dyn_press=self.rho*qe**2/2
        self.Red=self.rho*self.qe*self.delta/props.mu
        self.pp_w=clsr.ap_w+clsr.bp_w*self.Lambda
    def calc_derivs_x(self, dd_dx_seed=1.0, Ksi_mat_t1_arg=np.zeros((3, 3, 3, 3, 2))):
        dLambda_dx=(2*dd_dx_seed*self.drhoq_dx+self.delta*self.d2rhoq_dx2)*self.delta/self.atm_props.mu

        dRed_dx=(dd_dx_seed*self.rho*self.qe+self.delta*self.drhoq_dx)/self.atm_props.mu

        dUt_dx=-(dRed_dx*self.Ut**2*(1.0-np.cos(self.beta))+self.Red*self.Ut**2*np.sin(self.beta)*self.dbeta_dx+dLambda_dx*self.clsr.bp_w*self.C_Ut_dp\
            -self.pp_w*self.Ut**2*self.up_prime_edge*dRed_dx)/(2*self.Red*self.Ut*(1.0-np.cos(self.beta))-self.pp_w*(self.up_edge+self.Red*self.Ut*self.up_prime_edge))

        ddeltastar_dx=dRed_dx*self.Ut+self.Red*dUt_dx

        dC_dx=self.dC_dUt*dUt_dx+self.dC_ddp*ddeltastar_dx
        dtb_dx=self.dbeta_dx*(1.0+self.tanb**2)**2

        Ksi_mat_t1=np.copy(Ksi_mat_t1_arg)
        Ksi_mat_t1[:, :, :, :, 1]*=ddeltastar_dx

        Ksi_mat_t2=np.zeros((3, 3, 3, 2))
        Ksi_mat_t2[:, :, 0, :]=Ksi_mat_t1[:, :, 0, 0, :]
        Ksi_mat_t2[:, :, 1, 0]=Ksi_mat_t1[:, :, 1, 0, 0]+Ksi_mat_t1[:, :, 0, 1, 0]*self.Lambda
        Ksi_mat_t2[:, :, 1, 1]=Ksi_mat_t1[:, :, 1, 0, 1]+Ksi_mat_t1[:, :, 0, 1, 1]*self.Lambda+Ksi_mat_t1[:, :, 0, 1, 0]*dLambda_dx
        Ksi_mat_t2[:, :, 2, 0]=Ksi_mat_t1[:, :, 2, 0, 0]+2*Ksi_mat_t1[:, :, 1, 1, 0]*self.Lambda+Ksi_mat_t1[:, :, 0, 2, 0]*self.Lambda**2
        Ksi_mat_t2[:, :, 2, 1]=Ksi_mat_t1[:, :, 2, 0, 1]+2*(Ksi_mat_t1[:, :, 1, 1, 1]*self.Lambda+Ksi_mat_t1[:, :, 1, 1, 0]*dLambda_dx)+\
            Ksi_mat_t1[:, :, 0, 2, 1]*self.Lambda**2+2*Ksi_mat_t1[:, :, 0, 2, 0]*self.Lambda*dLambda_dx
        
        prods=np.zeros((3, 3, 3, 2))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    prods[i, j, k, 0]=self.tanb**i*self.Ut**j*self.C_Ut_dp**k
                    if i!=0:
                        prods[i, j, k, 1]+=i*self.tanb**(i-1)*self.Ut**j*self.C_Ut_dp**k*dtb_dx
                    if j!=0:
                        prods[i, j, k, 1]+=j*self.tanb**i*self.Ut**(j-1)*self.C_Ut_dp**k*dUt_dx
                    if k!=0:
                        prods[i, j, k, 1]+=k*self.tanb**i*self.Ut**j*self.C_Ut_dp**(k-1)*dC_dx
        
        dF_dx=deriv(Ksi_mat_t2, prods, prodinds=(0, 1, 0), Ksi_inds=(1, 0, 0))+deriv(Ksi_mat_t2, prods, prodinds=(0, 0, 1), Ksi_inds=(0, 0, 1))
        dG_dx=deriv(Ksi_mat_t2, prods, prodinds=(1, 1, 0), Ksi_inds=(1, 1, 0))+deriv(Ksi_mat_t2, prods, prodinds=(1, 0, 1), Ksi_inds=(0, 1, 1))
        dH_dx=deriv(Ksi_mat_t2, prods, prodinds=(1, 2, 0), Ksi_inds=(2, 1, 0))+2*deriv(Ksi_mat_t2, prods, prodinds=(1, 1, 1), Ksi_inds=(1, 1, 1))+\
            deriv(Ksi_mat_t2, prods, prodinds=(1, 0, 2), Ksi_inds=(0, 1, 2))
        dI_dx=deriv(Ksi_mat_t2, prods, prodinds=(0, 2, 0), Ksi_inds=(2, 0, 0))+2*deriv(Ksi_mat_t2, prods, prodinds=(0, 1, 1), Ksi_inds=(1, 0, 1))+\
            deriv(Ksi_mat_t2, prods, prodinds=(0, 0, 2), Ksi_inds=(0, 0, 2))
        dJ_dx=deriv(Ksi_mat_t2, prods, prodinds=(2, 2, 0), Ksi_inds=(2, 2, 0))+2*deriv(Ksi_mat_t2, prods, prodinds=(2, 1, 1), Ksi_inds=(1, 2, 1))+\
            deriv(Ksi_mat_t2, prods, prodinds=(2, 0, 2), Ksi_inds=(0, 2, 2))
        
        ddeltax_bar_dx=-dF_dx
        ddeltaz_bar_dx=dG_dx
        dThetaxx_dx=dF_dx-dI_dx
        dThetaxz_dx=dG_dx-dH_dx
        dThetazx_dx=dH_dx
        dThetazz_dx=dJ_dx

        #return: ddx_bar_deps, ddz, ddThetaxx...
        return dd_dx_seed*self.Thetaxx+self.delta*dThetaxx_dx, \
            dd_dx_seed*self.Thetaxz+self.delta*dThetaxz_dx, dd_dx_seed*self.Thetazx+self.delta*dThetazx_dx, dd_dx_seed*self.Thetazz+self.delta*dThetazz_dx
    def calc_derivs_z(self, dd_dz_seed=1.0, Ksi_mat_t1_arg=np.zeros((3, 3, 3, 3, 2))):
        dLambda_dz=(2*dd_dz_seed*self.drhoq_dx+self.delta*self.d2rhoq_dxdz)*self.delta/self.atm_props.mu

        dRed_dz=(dd_dz_seed*self.rho*self.qe+self.delta*self.drhoq_dz)/self.atm_props.mu

        dUt_dz=-(dRed_dz*self.Ut**2*(1.0-np.cos(self.beta))+self.Red*self.Ut**2*np.sin(self.beta)*self.dbeta_dz+dLambda_dz*self.clsr.bp_w*self.C_Ut_dp\
            -self.pp_w*self.Ut**2*self.up_prime_edge*dRed_dz)/(2*self.Red*self.Ut*(1.0-np.cos(self.beta))-self.pp_w*(self.up_edge+self.Red*self.Ut*self.up_prime_edge))

        ddeltastar_dz=dRed_dz*self.Ut+self.Red*dUt_dz

        dC_dz=self.dC_dUt*dUt_dz+self.dC_ddp*ddeltastar_dz
        dtb_dz=self.dbeta_dz*(1.0+self.tanb**2)**2

        Ksi_mat_t1=np.copy(Ksi_mat_t1_arg)
        Ksi_mat_t1[:, :, :, :, 1]*=ddeltastar_dz

        Ksi_mat_t2=np.zeros((3, 3, 3, 2))
        Ksi_mat_t2[:, :, 0, :]=Ksi_mat_t1[:, :, 0, 0, :]
        Ksi_mat_t2[:, :, 1, 0]=Ksi_mat_t1[:, :, 1, 0, 0]+Ksi_mat_t1[:, :, 0, 1, 0]*self.Lambda
        Ksi_mat_t2[:, :, 1, 1]=Ksi_mat_t1[:, :, 1, 0, 1]+Ksi_mat_t1[:, :, 0, 1, 1]*self.Lambda+Ksi_mat_t1[:, :, 0, 1, 0]*dLambda_dz
        Ksi_mat_t2[:, :, 2, 0]=Ksi_mat_t1[:, :, 2, 0, 0]+2*Ksi_mat_t1[:, :, 1, 1, 0]*self.Lambda+Ksi_mat_t1[:, :, 0, 2, 0]*self.Lambda**2
        Ksi_mat_t2[:, :, 2, 1]=Ksi_mat_t1[:, :, 2, 0, 1]+2*(Ksi_mat_t1[:, :, 1, 1, 1]*self.Lambda+Ksi_mat_t1[:, :, 1, 1, 0]*dLambda_dz)+\
            Ksi_mat_t1[:, :, 0, 2, 1]*self.Lambda**2+2*Ksi_mat_t1[:, :, 0, 2, 0]*self.Lambda*dLambda_dz
        
        prods=np.zeros((3, 3, 3, 2))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    prods[i, j, k, 0]=self.tanb**i*self.Ut**j*self.C_Ut_dp**k
                    if i!=0:
                        prods[i, j, k, 1]+=i*self.tanb**(i-1)*self.Ut**j*self.C_Ut_dp**k*dtb_dz
                    if j!=0:
                        prods[i, j, k, 1]+=j*self.tanb**i*self.Ut**(j-1)*self.C_Ut_dp**k*dUt_dz
                    if k!=0:
                        prods[i, j, k, 1]+=k*self.tanb**i*self.Ut**j*self.C_Ut_dp**(k-1)*dC_dz
        
        
        dF_dz=deriv(Ksi_mat_t2, prods, prodinds=(0, 1, 0), Ksi_inds=(1, 0, 0))+deriv(Ksi_mat_t2, prods, prodinds=(0, 0, 1), Ksi_inds=(0, 0, 1))
        dG_dz=deriv(Ksi_mat_t2, prods, prodinds=(1, 1, 0), Ksi_inds=(1, 1, 0))+deriv(Ksi_mat_t2, prods, prodinds=(1, 0, 1), Ksi_inds=(0, 1, 1))
        dH_dz=deriv(Ksi_mat_t2, prods, prodinds=(1, 2, 0), Ksi_inds=(2, 1, 0))+2*deriv(Ksi_mat_t2, prods, prodinds=(1, 1, 1), Ksi_inds=(1, 1, 1))+\
            deriv(Ksi_mat_t2, prods, prodinds=(1, 0, 2), Ksi_inds=(0, 1, 2))
        dI_dz=deriv(Ksi_mat_t2, prods, prodinds=(0, 2, 0), Ksi_inds=(2, 0, 0))+2*deriv(Ksi_mat_t2, prods, prodinds=(0, 1, 1), Ksi_inds=(1, 0, 1))+\
            deriv(Ksi_mat_t2, prods, prodinds=(0, 0, 2), Ksi_inds=(0, 0, 2))
        dJ_dz=deriv(Ksi_mat_t2, prods, prodinds=(2, 2, 0), Ksi_inds=(2, 2, 0))+2*deriv(Ksi_mat_t2, prods, prodinds=(2, 1, 1), Ksi_inds=(1, 2, 1))+\
            deriv(Ksi_mat_t2, prods, prodinds=(2, 0, 2), Ksi_inds=(0, 2, 2))
        
        dThetaxx_dz=dF_dz-dI_dz
        dThetaxz_dz=dG_dz-dH_dz
        dThetazx_dz=dH_dz
        dThetazz_dz=dJ_dz

        #return: ddx_bar_deps, ddz, ddThetaxx...
        return dd_dz_seed*self.Thetaxx+self.delta*dThetaxx_dz, dd_dz_seed*self.Thetaxz+self.delta*dThetaxz_dz, \
            dd_dz_seed*self.Thetazx+self.delta*dThetazx_dz, dd_dz_seed*self.Thetazz+self.delta*dThetazz_dz
    def calc_data(self, Ut_initguess=0.1):
        Ksi_mat_t1=self.turb_deduce(Ut_initguess=Ut_initguess)

        self.tanb=np.tan(self.beta)

        self.F=self.Ut*Ksi_mat_t1[1, 0, 0, 0, 0]+self.C_Ut_dp*(Ksi_mat_t1[0, 0, 1, 0, 0]+Ksi_mat_t1[0, 0, 0, 1, 0]*self.Lambda)
        self.G=self.tanb*(self.Ut*Ksi_mat_t1[1, 1, 0, 0, 0]+self.C_Ut_dp*(Ksi_mat_t1[0, 1, 1, 0, 0]+Ksi_mat_t1[0, 1, 0, 1, 0]*self.Lambda))
        self.H=self.tanb*(self.Ut**2*Ksi_mat_t1[2, 1, 0, 0, 0]+2*self.C_Ut_dp*self.Ut*(Ksi_mat_t1[1, 1, 1, 0, 0]+Ksi_mat_t1[1, 1, 0, 1, 0]*self.Lambda)+\
            self.C_Ut_dp**2*(Ksi_mat_t1[0, 1, 2, 0, 0]+2*self.Lambda*Ksi_mat_t1[0, 1, 1, 1, 0]+Ksi_mat_t1[0, 1, 0, 2, 0]*self.Lambda**2))
        self.I=self.Ut**2*Ksi_mat_t1[2, 0, 0, 0, 0]+2*self.C_Ut_dp*self.Ut*(Ksi_mat_t1[1, 0, 1, 0, 0]+Ksi_mat_t1[1, 0, 0, 1, 0]*self.Lambda)+\
            self.C_Ut_dp**2*(Ksi_mat_t1[0, 0, 2, 0, 0]+2*Ksi_mat_t1[0, 0, 1, 1, 0]*self.Lambda+Ksi_mat_t1[0, 0, 0, 2, 0]*self.Lambda**2)
        self.J=self.tanb**2*(Ksi_mat_t1[2, 2, 0, 0, 0]*self.Ut**2+2*self.C_Ut_dp*self.Ut*(Ksi_mat_t1[1, 2, 1, 0, 0]+Ksi_mat_t1[1, 2, 0, 1, 0]*self.Lambda)+\
            self.C_Ut_dp**2*(Ksi_mat_t1[0, 2, 2, 0, 0]+2*Ksi_mat_t1[0, 2, 1, 1, 0]*self.Lambda+Ksi_mat_t1[0, 2, 0, 2, 0]*self.Lambda**2))
        
        self.deltax_bar=1.0-self.F
        self.deltaz_bar=self.G
        self.Thetaxx=self.F-self.I
        self.Thetaxz=self.G-self.H
        self.Thetazx=self.H
        self.Thetazz=self.J

        return Ksi_mat_t1
    def turb_deduce(self, Ut_initguess=0.1):
        self.Ut=sopt.fsolve(self.turb_it, x0=Ut_initguess)[0]
        self.deltastar=self.Ut*self.Red
        self.Cf=self.Ut**2*2
        self.up_edge=self.clsr.LOTW(self.deltastar)
        self.C_Ut_dp=(1.0-self.Ut*self.up_edge)
        h=self.deltastar/self.clsr.Ksi_disc
        self.up_prime_edge=(self.clsr.LOTW(self.deltastar+h)-self.clsr.LOTW(self.deltastar-h))/(2*h)

        #obtain Ksis from closure relationships
        Ksi_W=self.clsr.Ksi_W(self.deltastar)
        dKsi_W_ddp=self.clsr.Ksi_W(self.deltastar, dx=1)
        Ksi_W2=self.clsr.Ksi_W2(self.deltastar)
        dKsi_W2_ddp=self.clsr.Ksi_W2(self.deltastar, dx=1)
        Ksi_Wa=self.clsr.Ksi_Wa(self.deltastar)
        dKsi_Wa_ddp=self.clsr.Ksi_Wa(self.deltastar, dx=1)
        Ksi_Wb=self.clsr.Ksi_Wb(self.deltastar)
        dKsi_Wb_ddp=self.clsr.Ksi_Wb(self.deltastar, dx=1)
        Ksi_WM=self.clsr.Ksi_WM(self.deltastar)
        dKsi_WM_ddp=self.clsr.Ksi_WM(self.deltastar, dx=1)
        Ksi_WMa=self.clsr.Ksi_WMa(self.deltastar)
        dKsi_WMa_ddp=self.clsr.Ksi_WMa(self.deltastar, dx=1)
        Ksi_WMb=self.clsr.Ksi_WMb(self.deltastar)
        dKsi_WMb_ddp=self.clsr.Ksi_WMb(self.deltastar, dx=1)
        Ksi_W2M=self.clsr.Ksi_W2M(self.deltastar)
        dKsi_W2M_ddp=self.clsr.Ksi_W2M(self.deltastar, dx=1)
        Ksi_W2M2=self.clsr.Ksi_W2M2(self.deltastar)
        dKsi_W2M2_ddp=self.clsr.Ksi_W2M2(self.deltastar, dx=1)
        Ksi_WM2a=self.clsr.Ksi_WM2a(self.deltastar)
        dKsi_WM2a_ddp=self.clsr.Ksi_WM2a(self.deltastar, dx=1)
        Ksi_WM2b=self.clsr.Ksi_WM2b(self.deltastar)
        dKsi_WM2b_ddp=self.clsr.Ksi_WM2b(self.deltastar, dx=1)

        #obtain ksis into matrix
        Ksi_mat_t1=np.zeros((3, 3, 3, 3, 2)) #into the matrix: indexes denote: W, M, a, b degree, and finally derivative index (0 for none, 1 for x, 2 for z)
        Ksi_mat_t1[1, 0, 0, 0, 0]=Ksi_W
        Ksi_mat_t1[1, 0, 0, 0, 1]=dKsi_W_ddp
        Ksi_mat_t1[2, 0, 0, 0, 0]=Ksi_W2
        Ksi_mat_t1[2, 0, 0, 0, 1]=dKsi_W2_ddp
        Ksi_mat_t1[1, 0, 1, 0, 0]=Ksi_Wa
        Ksi_mat_t1[1, 0, 1, 0, 1]=dKsi_Wa_ddp
        Ksi_mat_t1[1, 0, 0, 1, 0]=Ksi_Wb
        Ksi_mat_t1[1, 0, 0, 1, 1]=dKsi_Wb_ddp
        Ksi_mat_t1[1, 1, 0, 0, 0]=Ksi_WM
        Ksi_mat_t1[1, 1, 0, 0, 1]=dKsi_WM_ddp
        Ksi_mat_t1[1, 1, 1, 0, 0]=Ksi_WMa
        Ksi_mat_t1[1, 1, 1, 0, 1]=dKsi_WMa_ddp
        Ksi_mat_t1[1, 1, 0, 1, 0]=Ksi_WMb
        Ksi_mat_t1[1, 1, 0, 1, 1]=dKsi_WMb_ddp
        Ksi_mat_t1[2, 1, 0, 0, 0]=Ksi_W2M
        Ksi_mat_t1[2, 1, 0, 0, 1]=dKsi_W2M_ddp
        Ksi_mat_t1[2, 2, 0, 0, 0]=Ksi_W2M2
        Ksi_mat_t1[2, 2, 0, 0, 1]=dKsi_W2M2_ddp
        Ksi_mat_t1[1, 2, 1, 0, 0]=Ksi_WM2a
        Ksi_mat_t1[1, 2, 1, 0, 1]=dKsi_WM2a_ddp
        Ksi_mat_t1[1, 2, 0, 1, 0]=Ksi_WM2b
        Ksi_mat_t1[1, 2, 0, 1, 1]=dKsi_WM2b_ddp
        Ksi_mat_t1[0, 0, 1, 0, 0]=self.clsr.Ksi_a
        Ksi_mat_t1[0, 0, 0, 1, 0]=self.clsr.Ksi_b
        Ksi_mat_t1[0, 0, 1, 1, 0]=self.clsr.Ksi_ab
        Ksi_mat_t1[0, 0, 2, 0, 0]=self.clsr.Ksi_a2
        Ksi_mat_t1[0, 0, 0, 2, 0]=self.clsr.Ksi_b2
        Ksi_mat_t1[0, 1, 1, 0, 0]=self.clsr.Ksi_Ma
        Ksi_mat_t1[0, 1, 0, 1, 0]=self.clsr.Ksi_Mb
        Ksi_mat_t1[0, 1, 1, 1, 0]=self.clsr.Ksi_Mab
        Ksi_mat_t1[0, 1, 2, 0, 0]=self.clsr.Ksi_Ma2
        Ksi_mat_t1[0, 1, 0, 2, 0]=self.clsr.Ksi_Mb2
        Ksi_mat_t1[0, 2, 1, 1, 0]=self.clsr.Ksi_M2ab
        Ksi_mat_t1[0, 2, 2, 0, 0]=self.clsr.Ksi_M2a2
        Ksi_mat_t1[0, 2, 0, 2, 0]=self.clsr.Ksi_M2b2

        self.dC_dUt=-self.up_edge
        self.dC_ddp=-self.up_prime_edge*self.Ut
        return Ksi_mat_t1
    def turb_it(self, Ut): #function defining a single iteration for Newton's method
        return self.Red*Ut**2*(1.0-np.cos(self.beta))+self.pp_w*(1.0-Ut*self.clsr.LOTW(self.Red*Ut))
    def eqns_solve(self, Ksi_mat_t1=np.zeros((3, 3, 3, 3, 2))):
        #structure: A{dd_dx, dd_dz}.T+b={dthxx_dx+dthxz_dz, dthzx_dx+dthzz_dz}.T
        thxx_0x, thxz_0x, thzx_0x, thzz_0x=self.calc_derivs_x(dd_dx_seed=0.0, Ksi_mat_t1_arg=Ksi_mat_t1)
        thxx_0z, thxz_0z, thzx_0z, thzz_0z=self.calc_derivs_z(dd_dz_seed=0.0, Ksi_mat_t1_arg=Ksi_mat_t1)
        thxx_1x, thxz_1x, thzx_1x, thzz_1x=self.calc_derivs_x(dd_dx_seed=1.0, Ksi_mat_t1_arg=Ksi_mat_t1)
        thxx_1z, thxz_1z, thzx_1z, thzz_1z=self.calc_derivs_z(dd_dz_seed=1.0, Ksi_mat_t1_arg=Ksi_mat_t1)
        b=np.array([thxx_0x+thxz_0z, thzx_0x+thzz_0z])
        A=np.array([[thxx_1x-thxx_0x, thxz_1z-thxz_0z], [thzx_1x-thzx_0x, thzz_1z-thzz_0z]])
        RHS=np.array([np.cos(self.beta), np.sin(self.beta)])*self.Cf+np.array([-(self.deltax_bar*self.dq_dx+self.deltaz_bar*self.dq_dz), self.dq_dx*np.tan(self.beta)])*self.delta/self.qe-\
            np.array([self.Thetaxx*self.dq_dx+self.Thetaxz*self.dq_dz, self.Thetazx*self.dq_dx+self.Thetazz*self.dq_dz])*(2.0-self.Me**2)*self.delta/self.qe
        return lg.solve(A, RHS-b)
    def calcpropag(self, Ut_initguess=0.1):
        Ksi_mat_t1=self.calc_data(Ut_initguess=Ut_initguess)
        return self.eqns_solve(Ksi_mat_t1=Ksi_mat_t1)
