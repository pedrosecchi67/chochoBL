from chochoBL import *

import numpy as np
import time as tm

import pytest

def test_laminar_flat_plate():
    nm=100
    nn=50
    L=1.0

    Uinf=1.0

    xs=np.sin(np.pi*np.linspace(0.0, 1.0, nm)/2)**2*L
    ys=np.linspace(0.0, 1.0, nn)

    posits=np.zeros((nm, nn, 3))
    posaux=np.zeros((nm*nn, 3))

    n=0
    for i in range(nm):
        for j in range(nn):
            posits[i, j, 0]=xs[i]
            posits[i, j, 1]=ys[j]

            posaux[n, 0]=xs[i]
            posaux[n, 1]=ys[j]
            n+=1
    
    mu=defatm.mu
    rho0=defatm.rho

    th11=0.665*np.sqrt((mu*(posaux[:, 0]+1e-2))/(rho0*Uinf))
    H=2.5864*np.ones_like(th11)
    N=np.zeros_like(th11)
    
    vels=np.zeros((nm*nn, 3))
    vels[:, 0]=Uinf

    normals=np.zeros((nm*nn, 3))
    normals[:, 2]=1.0
    
    msh=mesh()

    inds=np.zeros((nm, nn), dtype='int')

    n=0
    for i in range(nm):
        for j in range(nn):
            msh.add_node(posits[i, j, :])

            inds[i, j]=n
            n+=1

    for i in range(nm-1):
        for j in range(nn-1):
            msh.add_cell({inds[i, j], inds[i, j+1], inds[i+1, j+1], inds[i+1, j]})
    
    msh.compose(normals)

    t=tm.time()

    msh.graph_init()

    msh.gr.heads['q'].set_value({'qx':vels[:, 0], 'qy':vels[:, 1], 'qz':vels[:, 2]})
    msh.gr.heads['th11'].set_value({'th11':th11})
    msh.gr.heads['H'].set_value({'H':H})
    msh.gr.heads['N'].set_value({'N':N})
    msh.gr.heads['beta'].set_value({'beta':np.zeros(nm*nn)})

    msh.gr.calculate(ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cf', 'Cd', 'J', 'M', 'E', 'rhoQ'])

    tdiff=tm.time()

    qx_derivs=msh.gr.get_derivs_direct('qx', ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cf', 'Cd', 'J', 'M', 'E', 'rhoQ'])
    #th22_derivs=msh.gr.get_derivs_reverse('thetastar_2', ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cf'])
    #deltaprime2_derivs=msh.gr.get_derivs_reverse('deltaprime_2', ends=['closure', 'p', 'uw', 'thetastar', 'deltaprime', 'Cf'])

    t=tm.time()-t

    tdiff=tm.time()-tdiff

    print(t, tdiff)

test_laminar_flat_plate()
