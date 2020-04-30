from chochoBL import *
import random as rnd
import numpy as np

def findiff_test():
    delta=rnd.random()
    drhoq_dx=0.0#rnd.random()
    drhoq_dz=0.0#rnd.random()
    d2rhoq_dx2=0.0#rnd.random()
    d2rhoq_dxdz=0.0#rnd.random()
    beta=0.0#rnd.random()
    dbeta_dx=0.0#rnd.random()
    dbeta_dz=0.0#rnd.random()
    qe=rnd.random()

    dd_dx_seed=rnd.random()
    dd_dz_seed=rnd.random()
    rho=defatm.rho

    h_x=1e-7
    h_z=1e-7

    stat=station(delta=delta, drhoq_dx=drhoq_dx, drhoq_dz=drhoq_dz, d2rhoq_dx2=d2rhoq_dx2, d2rhoq_dxdz=d2rhoq_dxdz, beta=beta, dbeta_dx=dbeta_dx, dbeta_dz=dbeta_dz, qe=qe)
    stat_dx=station(delta=delta+dd_dx_seed*h_x, drhoq_dx=drhoq_dx+d2rhoq_dx2*h_x, drhoq_dz=drhoq_dz+d2rhoq_dxdz*h_x, d2rhoq_dx2=d2rhoq_dx2, d2rhoq_dxdz=d2rhoq_dxdz, \
        beta=beta+dbeta_dx*h_x, dbeta_dx=dbeta_dx, dbeta_dz=dbeta_dz, qe=qe+drhoq_dx*h_x/rho)
    stat_dz=station(delta=delta+dd_dz_seed*h_z, drhoq_dx=drhoq_dx+d2rhoq_dxdz*h_z, drhoq_dz=drhoq_dz, d2rhoq_dx2=d2rhoq_dx2, d2rhoq_dxdz=d2rhoq_dxdz, \
        beta=beta+dbeta_dz*h_z, dbeta_dx=dbeta_dx, dbeta_dz=dbeta_dz, qe=qe+drhoq_dz*h_z/rho)
    
    stat.calc_data()
    stat_dx.calc_data()
    stat_dz.calc_data()

    _, _, thxx_an_x, _, _, _=stat.calc_derivs_x(dd_dx_seed)
    _, _, thxx_an_z, _, _, _=stat.calc_derivs_x(dd_dz_seed)

    dThxx_dx=(stat_dx.Thetaxx*stat_dx.delta-stat.Thetaxx*stat.delta)/h_x
    dThxx_dz=(stat_dz.Thetaxx*stat_dz.delta-stat.Thetaxx*stat.delta)/h_z
    print(abs(dThxx_dx-thxx_an_x)/dThxx_dx)
    print(abs(dThxx_dz-thxx_an_z)/dThxx_dz)

findiff_test()