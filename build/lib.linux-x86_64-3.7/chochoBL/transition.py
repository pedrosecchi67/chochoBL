import numpy as np
import scipy.sparse as sps
import scipy.special as special

from differentiation import *
from three_equation import _th11_tolerance

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

def dN_dReth(Rethc, sg, Reth, Hk, A, ismult=True):

    return 0.01*np.sqrt((2.4*Hk-3.7+2.5*np.tanh(1.5*Hk-4.65))**2+0.15)*(sg if ismult else 1.0)

def d2N_dReth2(dN_dR, dsg_dReth):
    '''
    Former parameter differentiated by Reth
    '''

    return dN_dR*dsg_dReth

def d2N_dHkdReth(Rethc, dRethc, sg, dsg_dHk, dN_dR, Reth, Hk, A):
    '''
    Former parameter differentiated by Hk
    '''

    return 0.01*sg*(2.4-3.75*(np.tanh(1.5*Hk-4.65)**2-1.0))*(2.4*Hk-3.7+2.5*np.tanh(1.5*Hk-4.65))/\
        np.sqrt((2.4*Hk-3.7+2.5*np.tanh(1.5*Hk-4.65))**2+0.15)+dN_dR*dsg_dHk

def _l(Hk):
    return (6.54*Hk-14.07)/Hk**2

def _dl_dHk(Hk):
    return -6.54/Hk**2+28.14/Hk**3

def _m(Hk):
    return (0.058*(Hk-4.0)**2/(Hk-1.0)-0.068)/_l(Hk)

def _dm_dHk(Hk):
    return 0.058*(Hk-4.0)*(Hk+2.0)/(Hk-1.0)**2/_l(Hk)-\
        (0.058*(Hk-4.0)**2/(Hk-1.0)-0.068)*_dl_dHk(Hk)/_l(Hk)**2

def p(dN_dR, th11, m, l):
    '''
    Parameter p(Hk, th11)=dN/ds
    '''

    return dN_dR*((m+1.0)/2)*l/th11

def dp_dHk(dN_dR, d2N_dHkdR, th11, m, dm, l, dl):
    '''
    Parameter p(Hk, th11)=dN/ds differentiated by density-independent shape 
    parameter Hk
    '''

    return sps.diags((d2N_dHkdR*((m+1.0)/2)*l+\
        dN_dR*dm*l/2+\
            dN_dR*((m+1.0)/2)*dl)/th11, format='lil')

def dp_dth11(pval, th11, passive):
    '''
    Parameter p(Hk, th11)=dN/ds differentiated by density-independent shape 
    parameter Hk
    '''

    return sps.diags(-pval/th11, format='lil')

def dp_dReth(d2N_dR2, m, l, th11):
    '''
    Parameter p(Hk, th11)=dN/ds differentiated by momentum thickness
    Reynolds number
    '''

    return sps.diags(d2N_dR2*((m+1.0)/2)*l/th11, format='lil')

def sigma_N(N, passive):
    '''
    Return sigma function of transition so that sigma_N*f_turbulent(...)+(1.0-sigma_N)*f_laminar(...)
    is a suitable transitional behavior approximation
    '''

    return _sigma((N-passive['Ncrit'])*passive['A_transition'])

def dsigma_N_dN(N, sn, passive):
    '''
    Return sigma function of transition, so that sigma_N*f_turbulent(...)+(1.0-sigma_N)*f_laminar(...)
    is a suitable transitional behavior approximation, differentiated by N
    '''

    A_transition=passive['A_transition']

    return sps.diags(A_transition*sn*(1.0-sn), format='lil')

def p_getnode(msh):
    '''
    Return a node for the function p(Reth, Hk, th11)=dN/ds
    '''

    def pfunc(Reth, Hk, th11, passive):
        A=passive['A_transition']

        Rethc=Reth_crit(Hk)
        dRethc=dReth_crit_dHk(Hk)

        sg=_sigma((Reth-Rethc)*A)
        dsg_loc=_dsigma_du((Reth-Rethc)*A)
        dsg_dHk=-A*dRethc*dsg_loc
        dsg_dReth=A*dsg_loc

        dNdR=dN_dReth(Rethc, sg, Reth, Hk, A, ismult=False)

        d2NdHkdR=d2N_dHkdReth(Rethc, dRethc, sg, dsg_dHk, dNdR, Reth, Hk, A)
        d2NdR2=d2N_dReth2(dNdR, dsg_dReth)

        m=_m(Hk)
        dm=_dm_dHk(Hk)

        l=_l(Hk)
        dl=_dl_dHk(Hk)

        dNdR*=sg

        th11_aux=th11.copy()
        th11_aux[th11_aux<_th11_tolerance]=_th11_tolerance

        pval=p(dNdR, th11_aux, m, l)

        dpdHk=dp_dHk(dNdR, d2NdHkdR, th11_aux, m, dm, l, dl)
        dpdR=dp_dReth(d2NdR2, m, l, th11_aux)
        dpdth11=dp_dth11(pval, th11_aux, passive)

        value={'p':pval}
        Jac={'p':{'Reth':dpdR, 'Hk':dpdHk, 'th11':dpdth11}}

        return value, Jac

    pnode=node(f=pfunc, args_to_inds=['Reth', 'Hk', 'th11'], outs_to_inds=['p'], passive=msh.passive, haspassive=True)

    return pnode

def sigma_N_getnode(msh):
    '''
    Return a node for sigma N prediction
    '''

    def sigmafunc(N, passive):
        sn=sigma_N(N, passive)

        value={'sigma_N':sn}
        Jac={'sigma_N':{'N':dsigma_N_dN(N, sn, passive)}}

        return value, Jac

    sigmanode=node(f=sigmafunc, args_to_inds=['N'], outs_to_inds=['sigma_N'], passive=msh.passive, haspassive=True)

    return sigmanode
