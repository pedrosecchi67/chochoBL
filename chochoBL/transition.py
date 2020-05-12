import numpy as np
from closure import *
from station import *

'''
Module containing functions necessary for transition prediction
'''

def Tollmien_Schlichting_Drela(stat):
    '''
    Returns the critical theta-measured Reynolds number for Tollmien-Schlichting waves as 
    a function of local Re_theta and H parameters (provided in station object stat).
    Drela and Giles's fit for Orr-Sommerfield equation eigenvalue solutions is used
    '''

    #gathering local shape parameter
    Hk=stat.dx/stat.th[0, 0]

    #deducing critical Reynolds number (in respect to theta) considering fit data
    log10Reth_critical=(1.415/(Hk-1.0)-0.489)*tanh(20.0/(Hk-1.0)-12.9)+3.295/(Hk-1.0)+0.44

    log10Reth_local=np.log10(stat.th[0, 0]*stat.qe*stat.rho/stat.atm_props.mu)

    return log10Reth_local>log10Reth_critical