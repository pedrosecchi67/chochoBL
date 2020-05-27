import numpy as np
import scipy.sparse as sps

'''
Module containing functions necessary for transition prediction as exposed by
Giles and Drela in their paper Viscous-Inviscid Analysis of Transonic and Low Reynolds
Number Airfoils (AAIA Journal, 1987)

sigmoids for transition approximation are here used for boolean approximation using parameter A
being sigma_A=sigma(A*x). A ought to be passed in passive argument dictionary
'''

def dN_dReth(Hk):
    return 0.01*np.sqrt((2.4*Hk-3.7+2.5*np.tanh(1.5*Hk-4.65))**2+0.15)

def d2N_dHkdReth(Hk):
    '''
    Former parameter differentiated by Hk
    '''

    return 0.01*(2.4-3.75*(np.tanh(1.5*Hk-4.65)**2-1.0))*(2.4*Hk-3.7+2.5*np.tanh(1.5*Hk-4.65))/\
        np.sqrt((2.4*Hk-3.7+2.5*np.tanh(1.5*Hk-4.65))**2+0.15)

def _l(Hk):
    return (6.54*Hk-14.07)/Hk**2

def _dl_dHk(Hk):
    return -6.54/Hk**2+28.14/Hk**3

def _m(Hk):
    return (0.058*(Hk-4.0)**2/(Hk-1.0)-0.068)/_l(Hk)

def _dm_dHk(Hk):
    return 0.058*(Hk-4.0)*(Hk+2.0)/(Hk-1.0)**2/_l(Hk)-\
        (0.058*(Hk-4.0)**2/(Hk-1.0)-0.068)*_dl_dHk(Hk)/_l(Hk)**2

def p(Hk, th11):
    '''
    Parameter p(Hk, th11)=dN/ds
    '''

    return dN_dReth(Hk)*((_m(Hk)+1.0)/2)*_l(Hk)/th11

def dp_dHk(Hk, th11):
    '''
    Parameter p(Hk, th11)=dN/ds differentiated by density-independent shape 
    parameter Hk
    '''

    return sps.diags((d2N_dHkdReth(Hk)*((_m(Hk)+1.0)/2)*_l(Hk)+\
        dN_dReth(Hk)*_dm_dHk(Hk)*_l(Hk)/2+\
            dN_dReth(Hk)*((_m(Hk)+1.0)/2)*_dl_dHk(Hk))/th11, format='lil')

def dp_th11(Hk, th11):
    '''
    Parameter p(Hk, th11)=dN/ds differentiated by density-independent shape 
    parameter Hk
    '''

    return sps.diags(-p(Hk, th11)/th11, format='lil')
