from chochoBL import *
import random as rnd
import numpy as np
import numpy.linalg as lg
from math import *

def findiff_test():
    delta=rnd.random()
    dq_dx=rnd.random()
    dq_dz=rnd.random()
    d2q_dx2=rnd.random()
    d2q_dxdz=rnd.random()
    beta=rnd.random()*0.2 #multiplying to mantain reasonable small-crossflow assumptions
    dbeta_dx=rnd.random()*0.1
    dbeta_dz=rnd.random()*0.1
    qe=rnd.random()

    dd_dx_seed=rnd.random()
    dd_dz_seed=rnd.random()
    rho=defatm.rho

    print('beta ', degrees(beta), ' dbeta_dx ', degrees(dbeta_dx), ' dbeta_dz ', degrees(dbeta_dz))
    print(dq_dx*np.tan(beta))

    h_x=1e-8
    h_z=1e-8

    stat=station(delta=delta, dq_dx=dq_dx, dq_dz=dq_dz, d2q_dx2=d2q_dx2, d2q_dxdz=d2q_dxdz, qe=qe, Uinf=qe, \
        beta=beta, dbeta_dx=dbeta_dx, dbeta_dz=dbeta_dz)
    stat_dx=station(delta=delta+dd_dx_seed*h_x, dq_dx=dq_dx+d2q_dx2*h_x, dq_dz=dq_dz+d2q_dxdz*h_x, \
        d2q_dx2=d2q_dx2, d2q_dxdz=d2q_dxdz, qe=qe+dq_dx*h_x, Uinf=qe, \
            beta=beta+dbeta_dx*h_x, dbeta_dx=dbeta_dx, dbeta_dz=dbeta_dz)
    stat_dz=station(delta=delta+dd_dz_seed*h_z, dq_dx=dq_dx+d2q_dxdz*h_z, dq_dz=dq_dz, \
        d2q_dx2=d2q_dx2, d2q_dxdz=d2q_dxdz, qe=qe+dq_dz*h_z, Uinf=qe, \
            beta=beta+dbeta_dz*h_z, dbeta_dx=dbeta_dx, dbeta_dz=dbeta_dz)
    
    stat.calc_data()
    stat_dx.calc_data()
    stat_dz.calc_data()

    dThxx_dx_an, dThxz_dx_an, dThzx_dx_an, dThzz_dx_an=stat.calc_derivs_x(dd_dx_seed)
    dThxx_dz_an, dThxz_dz_an, dThzx_dz_an, dThzz_dz_an=stat.calc_derivs_z(dd_dz_seed)

    devs=[]

    dThxx_dx=(stat_dx.Thetaxx*stat_dx.delta-stat.Thetaxx*stat.delta)/h_x
    dThxx_dz=(stat_dz.Thetaxx*stat_dz.delta-stat.Thetaxx*stat.delta)/h_z

    devs.append(abs((dThxx_dx-dThxx_dx_an)/dThxx_dx))
    devs.append(abs((dThxx_dz-dThxx_dz_an)/dThxx_dz))

    dThxz_dx=(stat_dx.Thetaxz*stat_dx.delta-stat.Thetaxz*stat.delta)/h_x
    dThxz_dz=(stat_dz.Thetaxz*stat_dz.delta-stat.Thetaxz*stat.delta)/h_z

    devs.append(abs((dThxz_dx-dThxz_dx_an)/dThxz_dx))
    devs.append(abs((dThxz_dz-dThxz_dz_an)/dThxz_dz))

    dThzx_dx=(stat_dx.Thetazx*stat_dx.delta-stat.Thetazx*stat.delta)/h_x
    dThzx_dz=(stat_dz.Thetazx*stat_dz.delta-stat.Thetazx*stat.delta)/h_z

    devs.append(abs((dThzx_dx-dThzx_dx_an)/dThzx_dx))
    devs.append(abs((dThzx_dz-dThzx_dz_an)/dThzx_dz))

    dThzz_dx=(stat_dx.Thetazz*stat_dx.delta-stat.Thetazz*stat.delta)/h_x
    dThzz_dz=(stat_dz.Thetazz*stat_dz.delta-stat.Thetazz*stat.delta)/h_z

    devs.append(abs((dThzz_dx-dThzz_dx_an)/dThzz_dx))
    devs.append(abs((dThzz_dz-dThzz_dz_an)/dThzz_dz))

    print(devs)

findiff_test()