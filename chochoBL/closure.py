import numpy as np
import scipy.optimize as sopt
import scipy.sparse as sps

from differentiation import *

Gersten_Herwig_A=6.1e-4
Gersten_Herwig_B=1.43e-3
Gersten_Herwig_Lambda=(Gersten_Herwig_A+Gersten_Herwig_B)**(1.0/3)
Von_Karman_kappa=0.41
def Gersten_Herwig_LOTW(yp):
    return (np.log((Gersten_Herwig_Lambda*yp+1.0)/np.sqrt((Gersten_Herwig_Lambda*yp)**2-\
        Gersten_Herwig_Lambda*yp+1.0))/3+(np.arctan((2*Gersten_Herwig_Lambda*yp-1.0)/np.sqrt(3))+np.pi/6)/np.sqrt(3))/Gersten_Herwig_Lambda+\
            np.log(1.0+Von_Karman_kappa*Gersten_Herwig_B*(yp)**4)/(4*Von_Karman_kappa)

'''
This module contains functions for modelling closure relations as exposed by
Giles and Drela in their paper Viscous-Inviscid Analysis of Transonic and Low Reynolds
Number Airfoils (AAIA Journal, 1987)
'''

def Hk(H, Me):
    '''
    Define density-independant equivalent shape parameter from compressible BL H shape parameter,
    as defined by Swafford
    '''

    return (H-0.290*Me**2)/(1.0+0.113*Me**2)

#last function's derivatives
def dHk_dH(H, Me):
    return sps.diags(1.0/(1.0+0.113*Me**2), format='lil')

def dHk_dMe(H, Me):
    return sps.diags(-((H-0.290*Me**2)*0.226*Me/(1.0+0.113*Me**2)+0.580*Me)/(1.0+0.113*Me**2), format='lil')

def Hk_getnode(msh):
    '''
    Returns a node containing a function for Hk computation, to be linked to H and Me nodes
    '''

    def Hkfunc(H, Me):
        value={'Hk':Hk(H, Me)}
        Jac={'Hk':{'H':dHk_dH(H, Me), 'Me':dHk_dMe(H, Me)}}

        return value, Jac
    
    Hknode=node(f=Hkfunc, args_to_inds=['H', 'Me'], outs_to_inds=['Hk'], passive=msh.passive)

    return Hknode

def _Hstar_laminar_attach(Hk):
    return 0.076*(4.0-Hk)**2/Hk+1.515

def _Hstar_laminar_detach(Hk):
    return 0.04*(Hk-4.0)**2/Hk+1.515

def _dHstar_laminar_dHk_attach(Hk):
    return 0.076-1.216/Hk**2

def _dHstar_laminar_dHk_detach(Hk):
    return 0.04-0.64/Hk**2

def Hstar_laminar(Hk):
    '''
    Return Hstar=thetastar/theta shape parameter
    '''

    attach=Hk<4.0
    detach=np.logical_not(attach)

    Hst=np.zeros_like(Hk)

    Hst[attach]=_Hstar_laminar_attach(Hk[attach])

    Hst[detach]=_Hstar_laminar_detach(Hk[detach])

    return Hst

def dHstar_laminar_dHk(Hk):
    '''
    Return the derivative of the Hstar=thetastar/theta shape parameter
    in respect to density-independent Hk shape parameter
    '''

    attach=Hk<4.0
    detach=np.logical_not(attach)

    Hst=np.zeros_like(Hk)

    Hst[attach]=_dHstar_laminar_dHk_attach(Hk[attach])

    Hst[detach]=_dHstar_laminar_dHk_detach(Hk[detach])

    return Hst

def _Tau_lowH(Hk):
    return 0.0396*(7.4-Hk)**2/(Hk-1.0)-0.134

def _dTau_lowH_dHk(Hk):
    return 0.0396*(Hk**2-2*Hk-39.96)/(Hk**2-2*Hk+1)

def _Tau_highH(Hk):
    return 0.044*(1.0-1.4/(Hk-6.0))**2-0.134

def _dTau_highH_dHk(Hk):
    return 0.1232*(Hk-7.4)/(Hk-6.0)**3

def _Tau(Hk):
    lowH=Hk<7.4
    highH=np.logical_not(lowH)

    T=np.zeros_like(Hk)

    T[lowH]=_Tau_lowH(Hk[lowH])

    T[highH]=_Tau_highH(Hk[highH])

    return T

def _dTau_dHk(Hk):
    lowH=Hk<7.4
    highH=np.logical_not(lowH)

    T=np.zeros_like(Hk)

    T[lowH]=_dTau_lowH_dHk(Hk[lowH])

    T[highH]=_dTau_highH_dHk(Hk[highH])

    return T

def Cf_laminar(T, Reth):
    '''
    Returns the streamwise friction coefficient according to Swafford-Giles-Drela\'s compressible 
    fit functions, for laminar regime
    '''

    return T/Reth

def dCf_laminar_dReth(T, Reth):
    '''
    Returns the derivative of the streamwise friction coefficient according to Swafford-Giles-Drela\'s
    compressible fit functions, for laminar regime, in respect to the momentum thickness Reynolds number
    '''

    return -T/Reth**2

def dCf_laminar_dHk(dT, Reth):
    '''
    Returns the derivative of the streamwise friction coefficient according to Swafford-Giles-Drela\'s
    compressible fit functions, for laminar regime, in respect to the density-independent compressible
    shape parameter Hk
    '''

    return dT/Reth

def Hprime_laminar(Me, Hk):
    '''
    Hprime shape factor (deltaprime/theta) for external Mach number Me and compressible, density-
    -independent shape parameter Hk
    '''

    return Me**2*(0.251+0.064/(Hk-2.5))

def dHprime_laminar_dMe(Me, Hk):
    '''
    Hprime shape factor (deltaprime/theta) derivative for external Mach number Me and compressible, 
    density-independent shape parameter Hk, in relationship to Me
    '''

    return 2*Me*(0.251+0.064/(Hk-2.5))

def dHprime_laminar_dHk(Me, Hk):
    '''
    Hprime shape factor (deltaprime/theta) derivative for external Mach number Me and compressible, 
    density-independent shape parameter Hk, in relationship to Hk
    '''

    return Me**2*(-0.064/(Hk-2.5)**2)

def _Delta_attached(Hk):
    return 0.0001025*(4.0-Hk)**5.5+0.1035

def _dDelta_attached_dHk(Hk):
    return -5.6375e-4*(4.0-Hk)**4.5

def _Delta_detached(Hk):
    return (0.207-0.003*(Hk-4.0)**2/(1.0+0.02*(Hk-4.0)**2))/2.0

def _dDelta_detached_dHk(Hk):
    return (-0.003*(Hk-4.0)/(1.0+0.02*(Hk-4.0)**2)**2)

def _Delta(Hk):
    attached=Hk<4.0
    detached=np.logical_not(attached)

    D=np.zeros_like(Hk)

    D[attached]=_Delta_attached(Hk[attached])
    D[detached]=_Delta_detached(Hk[detached])

    return D

def _dDelta_dHk(Hk):
    attached=Hk<4.0
    detached=np.logical_not(attached)

    D=np.zeros_like(Hk)

    D[attached]=_dDelta_attached_dHk(Hk[attached])
    D[detached]=_dDelta_detached_dHk(Hk[detached])

    return D

def Cd_laminar(D, Hst, Reth):
    '''
    Returns the dissipation coefficient according to Swafford-Giles-Drela compressible fit functions,
    given Reth, Me and Hk data
    '''

    return (D*Hst)/Reth

def dCd_laminar_dReth(Cd, Reth):
    '''
    Returns the dissipation coefficient according to Swafford-Giles-Drela compressible fit functions,
    given Reth and Hk data, derivated by momentum thickness Reynolds number
    '''

    return -Cd/Reth

def dCd_laminar_dHk(D, dD, Hst, dHst, Reth):
    '''
    Returns the dissipation coefficient according to Swafford-Giles-Drela compressible fit functions,
    given Reth and Hk data, derivated by external Mach number
    '''

    return (D*dHst+dD*Hst)/Reth

def _Hstar_Me0(Hk):
    return 1.81+3.84*np.exp(-2*Hk)-np.arctan((10.0**(7.0-Hk)-1.0)/1.23)/8.55-\
        0.146*np.sqrt(np.tanh(2.14*10.0**(4.0-1.46*Hk)))

def _dHstar_Me0_dHk(Hk):
    return -7.68*np.exp(-2*Hk)+1.87202*10.0**(7.0-Hk)/((((10.0**(7-Hk)-1.0)/1.23)**2+1.0)*8.55)+\
        1.05035*10.0**(4.0-1.46*Hk)*(1.0-np.tanh(2.14*10.0**(4-1.46*Hk))**2)/(2.0*np.sqrt(np.tanh(2.14*10.0**(4-1.46*Hk))))

def Hstar_turbulent(Hst_Me0, Me):
    '''
    Return turbulent shape factor Hstar (thetastar/theta)
    '''
    
    return (Hst_Me0+0.028*Me**2)/(1.0+0.014*Me**2)

def dHstar_turbulent_dHk(dHst_Me0_dHk, Me):
    '''
    Return turbulent shape factor Hstar (thetastar/theta) derivated by density independent shape
    parameter Hk
    '''

    return dHst_Me0_dHk/(1.0+0.014*Me**2)

def dHstar_turbulent_dMe(M0, Me):
    '''
    Return turbulent shape factor Hstar (thetastar/theta) derivated by external Mach number Me
    '''

    return 0.028*Me*(2.0-M0)/(0.014*Me**2+1.0)**2

def Hprime_turbulent(Me, Hk):
    '''
    Hprime turbulent shape factor (deltaprime (or deltastarstar)/theta)
    '''

    return Me**2*(0.251+0.064/(Hk-0.8))

def dHprime_turbulent_dMe(Me, Hk):
    '''
    Hprime turbulent shape factor (deltaprime (or deltastarstar)/theta) derivated by external 
    Mach number Me
    '''

    return Me*(0.502*Hk-0.2736)/(Hk-0.8)

def dHprime_turbulent_dHk(Me, Hk):
    '''
    Hprime turbulent shape factor (deltaprime (or deltastarstar)/theta) derivated by external 
    Mach number Me
    '''

    return -8.0*Me**2/(5.0*(5.0*Hk-4.0)**2)

def _Fc(Me, gamma):
    return np.sqrt(1.0+(gamma-1.0)*Me**2/2)

def _dFc_dMe(Me, gamma):
    return (gamma-1.0)*Me/(2.0*np.sqrt(1.0+(gamma-1.0)*Me**2/2))

def _f1(Reth, Hk):
    return 0.3*(np.log10(Reth))**(-0.31*Hk-1.74)*np.exp(-1.33*Hk)

def _df1_dHk(Reth, Hk):
    return -(0.093*np.log(np.log10(Reth))+0.399)*np.exp(-1.33*Hk)*np.log10(Reth)**(-0.31*Hk-1.74)

def _df1_dReth(Reth, Hk):
    return -(0.093*Hk+0.522)*np.exp(-1.33*Hk)*\
        np.log10(Reth)**(-0.31*Hk-1.74)/(Reth*np.log(Reth))

def _f2(Hk):
    return 0.00011*(np.tanh(4.0-8.0*Hk/7.0)-1.0)

def _df2_dHk(Hk):
    return 1.257e-4*(np.tanh(8.0*Hk/7.0-4.0)**2-1.0)

def _Cf_bar(Reth, Hk):
    return _f1(Reth, Hk)+_f2(Hk)

def _dCf_bar_dReth(Reth, Hk):
    return _df1_dReth(Reth, Hk)

def _dCf_bar_dHk(Reth, Hk):
    return _df1_dHk(Reth, Hk)+_df2_dHk(Hk)

def Cf_turbulent(Cf_b, F):
    '''
    Return turbulent friction coefficient
    '''

    return Cf_b/F

def dCf_turbulent_dReth(dCf_b_dR, F):
    '''
    Return derivative of turbulent friction coefficient in respect to momentum thickness
    Reynolds number
    '''

    return dCf_b_dR/F

def dCf_turbulent_dMe(Cf_b, F, dF_dMe):
    '''
    Return derivative of turbulent friction coefficient in respect to 
    externam Mach number Me
    '''

    return -dF_dMe*Cf_b/F

def dCf_turbulent_dHk(dCf_b_dHk, F):
    '''
    Return derivative of turbulent friction coefficient in respect to density-
    -independent shape parameter Hk
    '''

    return dCf_b_dHk/F

def _A(Hk):
    ishigh=Hk>3.5
    islow=np.logical_not(ishigh)

    answ=np.zeros_like(Hk)

    answ[ishigh]=0.160*(Hk[ishigh]-3.5)-0.550
    answ[islow]=0.438-0.280*Hk[islow]

    return answ

def _dA_dHk(Hk):
    ishigh=Hk>3.5
    islow=np.logical_not(ishigh)

    answ=np.zeros_like(Hk)

    answ[ishigh]=0.160
    answ[islow]=-0.280

    return answ

def _B(Hk):
    return 0.009-0.011*np.exp(-0.15*Hk**2.1)+3e-5*np.exp(0.117*Hk**2)

def _C(F, Me):
    return F*(1.0+0.05*Me**1.4)

def _dB_dHk(Hk):
    return 3.465e-3*Hk**1.1*np.exp(-0.15*Hk**2.1)+7.02e-6*Hk*np.exp(0.117*Hk**2)

def _dC_dMe(F, dF_dMe, Me):
    return dF_dMe*(1.0+0.05*Me**1.4)+F*0.07*Me**0.4

def _D(Reth):
    return Reth**(-0.574)

def _dD_dReth(Reth):
    return -0.574/Reth**1.574

def Cd_turbulent(A, B, C, D):
    '''
    Return turbulent dissipation coefficient
    '''

    return 2.0*(B+A*D)/C

def dCd_turbulent_dReth(A, C, dD_dR):
    '''
    Return turbulent dissipation coefficient differentiated by momentum thickness
    Reynolds number
    '''

    return 2.0*A*dD_dR/C

def dCd_turbulent_dMe(A, B, C, dC_dMe, D):
    '''
    Return turbulent dissipation coefficient
    '''

    return -2.0*(B+A*D)*dC_dMe/C**2

def dCd_turbulent_dHk(dA_dHk, dB_dHk, C, D):
    '''
    Return turbulent dissipation coefficient
    '''

    return 2.0*(dB_dHk+dA_dHk*D)/C

def _diag_lil(v):
    return sps.diags(v, format='lil')

def closure_getnode(msh):
    '''
    Return node for closure relationships, taking SG, Reth, Me and Hk and returning Hstar, 
    Hprime, Cf, Cd
    '''

    def fset(SG, Reth, Me, Hk, passive):
        Hst_lam=Hstar_laminar(Hk)
        dHst_lam_dHk=dHstar_laminar_dHk(Hk)

        Hpr_lam=Hprime_laminar(Me, Hk)
        dHpr_lam_dMe=dHprime_laminar_dMe(Me, Hk)
        dHpr_lam_dHk=dHprime_laminar_dHk(Me, Hk)

        T=_Tau(Hk)
        dT=_dTau_dHk(Hk)

        Cf_lam=Cf_laminar(T, Reth)
        dCf_lam_dHk=dCf_laminar_dHk(dT, Reth)
        dCf_lam_dR=dCf_laminar_dReth(T, Reth)

        D=_Delta(Hk)
        dD=_dDelta_dHk(Hk)

        Cd_lam=Cd_laminar(D, Hst_lam, Reth)
        dCd_lam_dR=dCd_laminar_dReth(Cd_lam, Reth)
        dCd_lam_dHk=dCd_laminar_dHk(D, dD, Hst_lam, dHst_lam_dHk, Reth)

        Hst_Me0=_Hstar_Me0(Hk)
        dHst_Me0_dHk=_dHstar_Me0_dHk(Hk)

        Hst_turb=Hstar_turbulent(Hst_Me0, Me)
        dHst_turb_dHk=dHstar_turbulent_dHk(dHst_Me0_dHk, Me)
        dHst_turb_dMe=dHstar_turbulent_dMe(Hst_Me0, Me)

        Hpr_turb=Hprime_turbulent(Me, Hk)
        dHpr_turb_dHk=dHprime_turbulent_dHk(Me, Hk)
        dHpr_turb_dMe=dHprime_turbulent_dMe(Me, Hk)

        gamma=passive['gamma']

        F=_Fc(Me, gamma)
        dF_dMe=_dFc_dMe(Me, gamma)

        Cf_b=_Cf_bar(Reth, Hk)
        dCf_b_dR=_dCf_bar_dReth(Reth, Hk)
        dCf_b_dHk=_dCf_bar_dHk(Reth, Hk)

        Cf_turb=Cf_turbulent(Cf_b, F)
        dCf_turb_dR=dCf_turbulent_dReth(dCf_b_dR, F)
        dCf_turb_dHk=dCf_turbulent_dHk(dCf_b_dHk, F)
        dCf_turb_dMe=dCf_turbulent_dMe(Cf_b, F, dF_dMe)

        A=_A(Hk)
        dA=_dA_dHk(Hk)
        B=_B(Hk)
        dB=_dB_dHk(Hk)

        C=_C(F, Me)
        dC_dMe=_dC_dMe(F, dF_dMe, Me)

        D=_D(Reth)
        dD_dR=_dD_dReth(Reth)

        Cd_turb=Cd_turbulent(A, B, C, D)
        dCd_turb_dR=dCd_turbulent_dReth(A, C, dD_dR)
        dCd_turb_dMe=dCd_turbulent_dMe(A, B, C, dC_dMe, D)
        dCd_turb_dHk=dCd_turbulent_dHk(dA, dB, C, D)

        SGC=(1.0-SG)

        value={
            'Hstar':Hst_lam*SGC+Hst_turb*SG,
            'Hprime':Hpr_lam*SGC+Hpr_turb*SG,
            'Cf':Cf_lam*SGC+Cf_turb*SG,
            'Cd':Cd_lam*SGC+Cd_turb*SG
        }

        Jac={
            'Hstar':{
                'sigma_N':_diag_lil(Hst_turb-Hst_lam),
                'Reth':None,
                'Me':_diag_lil(SG*dHst_turb_dMe),
                'Hk':_diag_lil(SGC*dHst_lam_dHk+SG*dHst_turb_dHk)
            },
            'Hprime':{
                'sigma_N':_diag_lil(Hpr_turb-Hpr_lam),
                'Reth':None,
                'Me':_diag_lil(SGC*dHpr_lam_dMe+SG*dHpr_turb_dMe),
                'Hk':_diag_lil(SGC*dHpr_lam_dHk+SG*dHpr_turb_dHk)
            },
            'Cf':{
                'sigma_N':_diag_lil(Cf_turb-Cf_lam),
                'Reth':_diag_lil(dCf_lam_dR*SGC+dCf_turb_dR*SG),
                'Me':_diag_lil(dCf_turb_dMe*SG),
                'Hk':_diag_lil(dCf_lam_dHk*SGC+dCf_turb_dHk*SG)
            },
            'Cd':{
                'sigma_N':_diag_lil(Cd_turb-Cd_lam),
                'Reth':_diag_lil(dCd_lam_dR*SGC+dCd_turb_dR*SG),
                'Me':_diag_lil(dCd_turb_dMe*SG),
                'Hk':_diag_lil(dCd_lam_dHk*SGC+dCd_turb_dHk*SG)
            }
        }

        return value, Jac

    newnode=node(f=fset, passive=msh.passive, args_to_inds=['sigma_N', 'Reth', 'Me', 'Hk'], outs_to_inds=['Hstar', 'Hprime', 'Cf', 'Cd'], \
        haspassive=True)

    return newnode

def f_crossflow(Cf, cosb, Me):
    return np.sqrt(Cf*cosb*(1.0+0.18*Me**2))

def df_crossflow_dCf(f, Cf):
    return f/(2*Cf)

def df_crossflow_dbeta(f, tb):
    return -tb*f/2

def df_crossflow_dMe(f, Me):
    return f*0.18*Me/(1.0+0.18*Me**2)

def g_crossflow(f):
    return f/(f-0.1)+1.0

def dg_crossflow_df(f):
    return -0.1/(f-0.1)**2

def A_crossflow(Cf, beta, Me):
    cosb=np.cos(beta)
    tb=np.tan(beta)

    isrev=Cf<0.0
    cosb[isrev]=-cosb[isrev]

    f=f_crossflow(Cf, cosb, Me)
    df_dCf=df_crossflow_dCf(f, Cf)
    df_dMe=df_crossflow_dMe(f, Me)
    df_dbeta=df_crossflow_dbeta(f, tb)

    g=g_crossflow(f)
    dg_df=dg_crossflow_df(f)

    A=tb*g
    dA_dbeta=g/cosb**2+dg_df*df_dbeta*tb
    dA_dCf=dg_df*tb*df_dCf
    dA_dMe=dg_df*tb*df_dMe

    return A, dA_dCf, dA_dbeta, dA_dMe, tb, 1/cosb**2

def A_crossflow_innode(Cf, beta, Me):
    A, dA_dCf, dA_dbeta, dA_dMe, tanb, dtanb_dbeta=A_crossflow(Cf, beta, Me)

    value={'A':A, 'tanb':tanb}
    Jac={
        'A':{
            'Cf':_diag_lil(dA_dCf),
            'beta':_diag_lil(dA_dbeta),
            'Me':_diag_lil(dA_dMe)
        },
        'tanb':{
            'Cf':None,
            'beta':_diag_lil(dtanb_dbeta),
            'Me':None
        }
    }

    return value, Jac

def A_getnode(msh):
    '''
    Obtain a node for A crossflow parameter
    '''

    A_node=node(f=A_crossflow_innode, args_to_inds=['Cf', 'beta', 'Me'], outs_to_inds=['A', 'tanb'], \
        passive=msh.passive)

    return A_node

def deltastar_innode(th11, H, A):
    deltastar_1=th11*H

    ddeltastar_1_dth11=H
    ddeltastar_1_dH=th11

    deltastar_2=-A*deltastar_1

    ddeltastar_2_dth11=-A*ddeltastar_1_dth11
    ddeltastar_2_dH=-A*ddeltastar_1_dH
    ddeltastar_2_dA=-deltastar_1

    value={'deltastar_1':deltastar_1, 'deltastar_2':deltastar_2}
    Jac={
        'deltastar_1':{
            'th11':_diag_lil(ddeltastar_1_dth11),
            'H':_diag_lil(ddeltastar_1_dH),
            'A':None
        },
        'deltastar_2':{
            'th11':_diag_lil(ddeltastar_2_dth11),
            'H':_diag_lil(ddeltastar_2_dH),
            'A':_diag_lil(ddeltastar_2_dA)
        }
    }

    return value, Jac

def deltastar_getnode(msh):
    '''
    Obtain a node for deltastar displacement thickness calculation
    '''

    dnode=node(f=deltastar_innode, args_to_inds=['th11', 'H', 'A'], outs_to_inds=['deltastar_1', 'deltastar_2'], \
        passive=msh.passive)

    return dnode

def Cf_innode(Cf, tanb):
    Cf_2=Cf*tanb

    dCf_2_dtanb=Cf
    dCf_2_dCf=tanb

    value={'Cf_2':Cf_2}
    Jac={
        'Cf_2':{
            'Cf':_diag_lil(dCf_2_dCf),
            'tanb':_diag_lil(dCf_2_dtanb)
        }
    }

    return value, Jac

def Cf_getnode(msh):
    '''
    Obtain a node for crossflow friction coefficient
    '''

    cfnode=node(f=Cf_innode, args_to_inds=['Cf', 'tanb'], outs_to_inds=['Cf_2'], passive=msh.passive)

    return cfnode

def theta_innode(th11, A, deltastar_2):
    th21=-A*th11
    th12=th21-deltastar_2
    th22=-A*th12

    dth21_dA=-th11
    dth21_dth11=-A

    dth12_dA=dth21_dA
    dth12_dth11=dth21_dth11
    dth12_ddeltastar_2=-np.ones_like(deltastar_2)

    dth22_dA=-(th12+dth12_dA*A)
    dth22_dth11=-A*dth12_dth11
    dth22_ddeltastar_2=-A*dth12_ddeltastar_2

    value={'th12':th12, 'th21':th21, 'th22':th22}
    Jac={
        'th12':{
            'th11':_diag_lil(dth12_dth11),
            'A':_diag_lil(dth12_dA),
            'deltastar_2':_diag_lil(dth12_ddeltastar_2)
        },
        'th21':{
            'th11':_diag_lil(dth21_dth11),
            'A':_diag_lil(dth21_dA),
            'deltastar_2':None
        },
        'th22':{
            'th11':_diag_lil(dth22_dth11),
            'A':_diag_lil(dth22_dA),
            'deltastar_2':_diag_lil(dth22_ddeltastar_2)
        }
    }

    return value, Jac

def theta_getnode(msh):
    '''
    Obtains a node to calculate theta momentum thicknesses for a mesh
    '''

    thnode=node(f=theta_innode, args_to_inds=['th11', 'A', 'deltastar_2'], outs_to_inds=['th12', 'th21', 'th22'])

    return thnode

def thetastar_innode(Hstar, A, deltastar_1, th11, th22):
    thetastar_1=Hstar*th11

    parc=(deltastar_1+th11+th22-thetastar_1)
    thetastar_2=A*parc

    value={'thetastar_1':thetastar_1, 'thetastar_2':thetastar_2}

    Jac={
        'thetastar_1':{
            'Hstar':_diag_lil(th11),
            'A':None,
            'deltastar_1':None,
            'th11':_diag_lil(Hstar),
            'th22':None
        },
        'thetastar_2':{
            'Hstar':_diag_lil(-A*th11),
            'A':_diag_lil(parc),
            'deltastar_1':_diag_lil(A),
            'th11':_diag_lil(A-A*Hstar),
            'th22':_diag_lil(A)
        }
    }

    return value, Jac

def thetastar_getnode(msh):
    '''
    Returns a node to calculate thetastar within a mesh
    '''

    thstnode=node(f=thetastar_innode, args_to_inds=['Hstar', 'A', 'deltastar_1', 'th11', 'th22'], outs_to_inds=['thetastar_1', 'thetastar_2'], \
        passive=msh.passive)

    return thstnode
