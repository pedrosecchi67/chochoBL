from chochoBL import *

import numpy as np
import time as tm

import pytest

def test_laminar_flat_plate():
    nm=5
    nn=2
    L=1.0

    Uinf=1.0

    xs=np.sin(np.pi*np.linspace(0.0, 1.0, nm)/2)**2*L
    ys=np.linspace(0.0, 1.0, nn)

    posits=np.zeros((nm, nn, 3))
    posaux=np.zeros((nm*nn, 3))

    vels=np.zeros((nm*nn, 3))

    n=0
    for i in range(nm):
        for j in range(nn):
            posits[i, j, 0]=xs[i]
            posits[i, j, 1]=ys[j]

            posaux[n, 0]=xs[i]
            posaux[n, 1]=ys[j]

            vels[n, 0]=Uinf*(xs[i]-0.5*L)

            n+=1
    
    mu=defatm.mu
    rho0=defatm.rho

    th11=0.29235*np.sqrt(mu/(rho0*Uinf))*np.ones(nm*nn)
    H=2.21622*np.ones_like(th11)
    N=np.zeros_like(th11)
    nflow=np.zeros_like(th11)

    normals=np.zeros((nm*nn, 3))
    normals[:, 2]=1.0
    
    msh=mesh(Uinf=Uinf)

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

    msh.graph_init()

    t=tm.time()

    vals={
        'q':{'qx':vels[:, 0], 'qy':vels[:, 1], 'qz':vels[:, 2]}, 
        'th11':{'th11':th11},
        'H':{'H':H},
        'N':{'N':N},
        'beta':{'beta':np.zeros(nm*nn)},
        'n':{'n':nflow}
    }

    msh.set_values(vals)

    value, grad=msh.calculate_graph()

    t=tm.time()-t

    print(t)

    print(value, grad)

test_laminar_flat_plate()
