import numpy as np
import scipy.optimize as sopt
import scipy.sparse as sps

import abaqus as abq

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
    return sps.diags(1.0/(1.0+0.113*Me**2))

def dHk_dMe(H, Me):
    return sps.diags(-2.0*Me*(0.113*H+0.290)/(0.113*Me**2+1.0))

def _Hstar_laminar_attach(Hk):
    return (0.076*(4.0-Hk)**2+1.515)/Hk

def _Hstar_laminar_detach(Hk):
    return (0.04*(Hk-4.0)**2+1.515)/Hk

def _dHstar_laminar_dHk_attach(Hk):
    return 0.076-2.731/Hk**2

def _dHstar_laminar_dHk_detach(Hk):
    return 0.04-2.155/Hk**2

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

def Cf_laminar(Reth, Hk):
    '''
    Returns the streamwise friction coefficient according to Swafford-Giles-Drela\'s compressible 
    fit functions, for laminar regime
    '''

    T=_Tau(Hk)

    return T/Reth

def dCf_laminar_dReth(Reth, Hk):
    '''
    Returns the derivative of the streamwise friction coefficient according to Swafford-Giles-Drela\'s
    compressible fit functions, for laminar regime, in respect to the momentum thickness Reynolds number
    '''

    T=_Tau(Hk)

    return -T/Reth**2

def dCf_laminar_dHk(Reth, Hk):
    '''
    Returns the derivative of the streamwise friction coefficient according to Swafford-Giles-Drela\'s
    compressible fit functions, for laminar regime, in respect to the density-independent compressible
    shape parameter Hk
    '''

    return _dTau_dHk(Hk)/Reth

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

    return Me*(0.502*Hk-0.2736)/(Hk-0.8)

def dHprime_laminar_dHk(Me, Hk):
    '''
    Hprime shape factor (deltaprime/theta) derivative for external Mach number Me and compressible, 
    density-independent shape parameter Hk, in relationship to Hk
    '''

    return -0.32*Me**2/(Hk-0.8)**2

def _Delta_attached(Hk):
    return 0.0001025*(4.0-Hk)**5.5+0.1035

def _dDelta_attached_dHk(Hk):
    return -0.00564*(4.0-Hk)**4.5

def _Delta_detached(Hk):
    return -3.0*(Hk-4.0)**2/(40.0*(Hk-4.0)**2+2000.0)+0.1035

def _dDelta_detached_dHk(Hk):
    return (30.0-7.5*Hk)/(Hk**4-16.0*Hk**3+196*Hk**2-1056*Hk**2+4356)

def Cd_laminar(Reth, Hk):
    '''
    Returns the dissipation coefficient according to Swafford-Giles-Drela compressible fit functions,
    given Reth, Me and Hk data
    '''

    attach=Hk<4.0
    detach=np.logical_not(attach)

    coef=np.zeros_like(Reth)

    coef[attach]=_Delta_attached(Hk[attach])
    coef[detach]=_Delta_detached(Hk[detach])

    par=Hstar_laminar(Hk)

    return (coef*par)/Reth

def dCd_laminar_dReth(Reth, Hk):
    '''
    Returns the dissipation coefficient according to Swafford-Giles-Drela compressible fit functions,
    given Reth and Hk data, derivated by momentum thickness Reynolds number
    '''

    coef=Cd_laminar(Reth, Me, Hk)

    return -coef/Reth

def dCd_laminar_dHk(Reth, Hk):
    '''
    Returns the dissipation coefficient according to Swafford-Giles-Drela compressible fit functions,
    given Reth and Hk data, derivated by external Mach number
    '''

    attach=Hk<4.0
    detach=np.logical_not(attach)

    coef=np.zeros_like(Reth)
    dcoef=np.zeros_like(Reth)

    dcoef[attach]=_dDelta_attached_dHk(Hk[attach])
    dcoef[detach]=_dDelta_detached_dHk(Hk[detach])

    coef[attach]=_Delta_attached(Hk[attach])
    coef[detach]=_Delta_detached(Hk[detach])

    par=Hstar_laminar(Hk)
    dpar=dHstar_laminar_dHk(Hk)

    return (coef*dpar+dcoef*par)/Reth
