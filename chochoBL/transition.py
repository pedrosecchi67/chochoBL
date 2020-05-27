import numpy as np
import scipy.sparse as sps
import scipy.special as special

from differentiation import *

'''
Module containing functions necessary for transition prediction as exposed by
Giles and Drela in their paper Viscous-Inviscid Analysis of Transonic and Low Reynolds
Number Airfoils (AAIA Journal, 1987)

sigmoids for transition approximation are here used for boolean approximation using parameter A
being sigma_A=sigma(A*x). A ought to be passed in passive argument dictionary
'''

def _sigma(u):
    return special.expit(u)

def _dsigma_du(u):
    return special.expit(u)*(1.0-special.expit(u))

def _A(Hk):
    return (1.415/(Hk-1.0)-0.489)*np.tanh(20.0/(Hk-1.0)-12.9)

def _dA_dHk(Hk):
    th=np.tanh(12.9+20.0/(1.0-Hk))

    return (1.415*(1.0-Hk)*th-\
        (28.30+9.78*(1.0-Hk))*(th**2-1.0))/(1.0-Hk)**3

def _B(Hk):
    return 3.295/(Hk-1.0)+0.44

def _dB_dHk(Hk):
    return -3.295/(Hk-1.0)**2

def _log10Reth_crit(Hk):
    return (_A(Hk)+_B(Hk))

def _dlog10Reth_crit_dHk(Hk):
    return _dA_dHk(Hk)+_dB_dHk(Hk)

def Reth_crit(Hk):
    '''
    Return critical Reynolds number in respect to momentum thickness for given density-independent
    shape parameter Hk
    '''

    return 10.0**_log10Reth_crit(Hk)

def dReth_crit_dHk(Hk):
    '''
    Return critical Reynolds number in respect to momentum thickness for given density-independent
    shape parameter Hk
    '''

    return Reth_crit(Hk)*np.log(10.0)*_dlog10Reth_crit_dHk(Hk)

def dN_dReth(Reth, Hk, A, ismult=True):
    Rethc=Reth_crit(Hk)
    
    sg=_sigma((Reth-Rethc)*A)

    return 0.01*np.sqrt((2.4*Hk-3.7+2.5*np.tanh(1.5*Hk-4.65))**2+0.15)*(sg if ismult else 1.0)

def d2N_dReth2(Reth, Hk, A):
    '''
    Former parameter differentiated by Reth
    '''

    Rethc=Reth_crit(Hk)

    dsg=A*_dsigma_du((Reth-Rethc)*A)

    return dN_dReth(Reth, Hk, A, ismult=False)*dsg

def d2N_dHkdReth(Reth, Hk, A):
    '''
    Former parameter differentiated by Hk
    '''

    Rethc=Reth_crit(Hk)
    dRethc=dReth_crit_dHk(Hk)

    sg=_sigma((Reth-Rethc)*A)
    dsg=-A*dRethc*_dsigma_du((Reth-Rethc)*A)

    return 0.01*sg*(2.4-3.75*(np.tanh(1.5*Hk-4.65)**2-1.0))*(2.4*Hk-3.7+2.5*np.tanh(1.5*Hk-4.65))/\
        np.sqrt((2.4*Hk-3.7+2.5*np.tanh(1.5*Hk-4.65))**2+0.15)+dN_dReth(Reth, Hk, A, ismult=False)*dsg

def _l(Hk):
    return (6.54*Hk-14.07)/Hk**2

def _dl_dHk(Hk):
    return -6.54/Hk**2+28.14/Hk**3

def _m(Hk):
    return (0.058*(Hk-4.0)**2/(Hk-1.0)-0.068)/_l(Hk)

def _dm_dHk(Hk):
    return 0.058*(Hk-4.0)*(Hk+2.0)/(Hk-1.0)**2/_l(Hk)-\
        (0.058*(Hk-4.0)**2/(Hk-1.0)-0.068)*_dl_dHk(Hk)/_l(Hk)**2

def p(Reth, Hk, th11, passive):
    '''
    Parameter p(Hk, th11)=dN/ds
    '''

    A=passive['A_Rethcrit']

    return dN_dReth(Reth, Hk, A)*((_m(Hk)+1.0)/2)*_l(Hk)/th11

def dp_dHk(Reth, Hk, th11, passive):
    '''
    Parameter p(Hk, th11)=dN/ds differentiated by density-independent shape 
    parameter Hk
    '''

    A=passive['A_Rethcrit']

    return sps.diags((d2N_dHkdReth(Reth, Hk, A)*((_m(Hk)+1.0)/2)*_l(Hk)+\
        dN_dReth(Reth, Hk, A)*_dm_dHk(Hk)*_l(Hk)/2+\
            dN_dReth(Reth, Hk, A)*((_m(Hk)+1.0)/2)*_dl_dHk(Hk))/th11, format='lil')

def dp_dth11(Reth, Hk, th11, passive):
    '''
    Parameter p(Hk, th11)=dN/ds differentiated by density-independent shape 
    parameter Hk
    '''

    return sps.diags(-p(Reth, Hk, th11, passive)/th11, format='lil')

def dp_dReth(Reth, Hk, th11, passive):
    '''
    Parameter p(Hk, th11)=dN/ds differentiated by momentum thickness
    Reynolds number
    '''

    A=passive['A_Rethcrit']

    return sps.diags(d2N_dReth2(Reth, Hk, A)*((_m(Hk)+1.0)/2)*_l(Hk)/th11, format='lil')

def sigma_N(N, passive):
    '''
    Return sigma function of transition so that sigma_N*f_turbulent(...)+(1.0-sigma_N)*f_laminar(...)
    is a suitable transitional behavior approximation
    '''

    return 1.0/(np.exp((passive['Ncrit']-N)*passive['A_transition'])+1.0)

def dsigma_N_dN(N, passive):
    '''
    Return sigma function of transition, so that sigma_N*f_turbulent(...)+(1.0-sigma_N)*f_laminar(...)
    is a suitable transitional behavior approximation, differentiated by N
    '''

    A_transition=passive['A_transition']

    E=np.exp((passive['Ncrit']-N)*A_transition)

    return sps.diags(A_transition*E/(E+1.0)**2, format='lil')

def p_getnode(msh):
    '''
    Return a node for the function p(Reth, Hk, th11)=dN/ds
    '''

    pfunc=func(f=p, derivs=(dp_dReth, dp_dHk, dp_dth11,), args=[0, 1, 2], sparse=True, haspassive=True)

    pnode=node(f=pfunc, args_to_inds=['Reth', 'Hk', 'th11'], outs_to_inds=['p'], passive=msh.passive)

    return pnode
