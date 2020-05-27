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

    Hkfunc=func(f=Hk, args=[0, 1], derivs=(dHk_dH, dHk_dMe,), sparse=True, haspassive=False)

    Hknode=node(f=Hkfunc, args_to_inds=['H', 'Me'], outs_to_inds=['Hk'], passive=msh.passive)

    return Hknode

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

    return sps.diags(Hst, format='lil')

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

    T=-_Tau(Hk)

    return sps.diags(T/Reth**2, format='lil')

def dCf_laminar_dHk(Reth, Hk):
    '''
    Returns the derivative of the streamwise friction coefficient according to Swafford-Giles-Drela\'s
    compressible fit functions, for laminar regime, in respect to the density-independent compressible
    shape parameter Hk
    '''

    return sps.diags(_dTau_dHk(Hk)/Reth, format='lil')

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

    return sps.diags(2*Me*(0.251+0.064/(Hk-2.5)), format='lil')

def dHprime_laminar_dHk(Me, Hk):
    '''
    Hprime shape factor (deltaprime/theta) derivative for external Mach number Me and compressible, 
    density-independent shape parameter Hk, in relationship to Hk
    '''

    return sps.diags(Me**2*(-0.064/(Hk-2.5)**2), format='lil')

def _Delta_attached(Hk):
    return 0.0001025*(4.0-Hk)**5.5+0.1035

def _dDelta_attached_dHk(Hk):
    return -5.6375e-4*(4.0-Hk)**4.5

def _Delta_detached(Hk):
    return (0.207-0.003*(Hk-4.0)**2/(1.0+0.02*(Hk-4.0)**2))/2.0

def _dDelta_detached_dHk(Hk):
    return (-0.003*(Hk-4.0)/(1.0+0.02*(Hk-4.0)**2)**2)

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

    coef=-Cd_laminar(Reth, Hk)

    return sps.diags(coef/Reth, format='lil')

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

    return sps.diags((coef*dpar+dcoef*par)/Reth, format='lil')

def _Hstar_Me0(Hk):
    return 1.81+3.84*np.exp(-2*Hk)-np.arctan((10.0**(7.0-Hk)-1.0)/1.23)/8.55-\
        0.146*np.sqrt(np.tanh(2.14*10.0**(4.0-1.46*Hk)))

def _dHstar_Me0_dHk(Hk):
    return -7.68*np.exp(-2*Hk)+1.87202*10.0**(7.0-Hk)/((((10.0**(7-Hk)-1.0)/1.23)**2+1.0)*8.55)+\
        1.05035*10.0**(4.0-1.46*Hk)*(1.0-np.tanh(2.14*10.0**(4-1.46*Hk))**2)/(2.0*np.sqrt(np.tanh(2.14*10.0**(4-1.46*Hk))))

def Hstar_turbulent(Me, Hk):
    '''
    Return turbulent shape factor Hstar (thetastar/theta)
    '''
    
    return (_Hstar_Me0(Hk)+0.028*Me**2)/(1.0+0.014*Me**2)

def dHstar_turbulent_dHk(Me, Hk):
    '''
    Return turbulent shape factor Hstar (thetastar/theta) derivated by density independent shape
    parameter Hk
    '''

    dM0=_dHstar_Me0_dHk(Hk)

    return sps.diags(dM0/(1.0+0.014*Me**2), format='lil')

def dHstar_turbulent_dMe(Me, Hk):
    '''
    Return turbulent shape factor Hstar (thetastar/theta) derivated by external Mach number Me
    '''
    
    M0=_Hstar_Me0(Hk)

    return sps.diags(0.028*Me*(2.0-M0)/(0.014*Me**2+1.0)**2, format='lil')

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

    return sps.diags(Me*(0.502*Hk-0.2736)/(Hk-0.8), format='lil')

def dHprime_turbulent_dHk(Me, Hk):
    '''
    Hprime turbulent shape factor (deltaprime (or deltastarstar)/theta) derivated by external 
    Mach number Me
    '''

    return sps.diags(-8.0*Me**2/(5.0*(5.0*Hk-4.0)**2), format='lil')

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

def Cf_turbulent(Reth, Me, Hk, passive):
    '''
    Return turbulent friction coefficient
    '''

    return _Cf_bar(Reth, Hk)/_Fc(Me, passive['gamma'])

def dCf_turbulent_dReth(Reth, Me, Hk, passive):
    '''
    Return derivative of turbulent friction coefficient in respect to momentum thickness
    Reynolds number
    '''

    gamma=passive['gamma']

    return sps.diags(_dCf_bar_dReth(Reth, Hk)/_Fc(Me, gamma), format='lil')

def dCf_turbulent_dMe(Reth, Me, Hk, passive):
    '''
    Return derivative of turbulent friction coefficient in respect to 
    externam Mach number Me
    '''

    gamma=passive['gamma']

    return sps.diags(-_dFc_dMe(Me, gamma)*_Cf_bar(Reth, Hk)/_Fc(Me, gamma)**2, format='lil')

def dCf_turbulent_dHk(Reth, Me, Hk, passive):
    '''
    Return derivative of turbulent friction coefficient in respect to density-
    -independent shape parameter Hk
    '''

    gamma=passive['gamma']

    return sps.diags(_dCf_bar_dHk(Reth, Hk)/_Fc(Me, gamma), format='lil')

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

def _C(Me, gamma):
    return _Fc(Me, gamma)*(1.0+0.05*Me**1.4)

def _dB_dHk(Hk):
    return 3.465e-3*Hk**1.1*np.exp(-0.15*Hk**2.1)+7.02e-6*Hk*np.exp(0.117*Hk**2)

def _dC_dMe(Me, gamma):
    return _dFc_dMe(Me, gamma)*(1.0+0.05*Me**1.4)+_Fc(Me, gamma)*0.07*Me**0.4

def _D(Reth):
    return Reth**(-0.574)

def _dD_dReth(Reth):
    return -0.574/Reth**1.574

def Cd_turbulent(Reth, Me, Hk, passive):
    '''
    Return turbulent dissipation coefficient
    '''

    gamma=passive['gamma']

    return 2.0*(_B(Hk)+_A(Hk)*_D(Reth))/_C(Me, gamma)

def dCd_turbulent_dReth(Reth, Me, Hk, passive):
    '''
    Return turbulent dissipation coefficient differentiated by momentum thickness
    Reynolds number
    '''

    gamma=passive['gamma']

    return sps.diags(2.0*_A(Hk)*_dD_dReth(Reth)/_C(Me, gamma), format='lil')

def dCd_turbulent_dMe(Reth, Me, Hk, passive):
    '''
    Return turbulent dissipation coefficient
    '''

    gamma=passive['gamma']

    return sps.diags(-2.0*(_B(Hk)+_A(Hk)*_D(Reth))*_dC_dMe(Me, gamma)/_C(Me, gamma)**2, format='lil')

def dCd_turbulent_dHk(Reth, Me, Hk, passive):
    '''
    Return turbulent dissipation coefficient
    '''

    gamma=passive['gamma']

    return sps.diags(2.0*(_dB_dHk(Hk)+_dA_dHk(Hk)*_D(Reth))/_C(Me, gamma), format='lil')
