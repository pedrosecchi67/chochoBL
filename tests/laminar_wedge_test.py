from chochoBL import *

import numpy as np
import numpy.linalg as lg
import scipy.optimize as sopt
import time as tm

import matplotlib.pyplot as plt

import pytest

def test_laminar_wedge():
    '''
    Testing Garlekin solution for an alpha=0.0 wedge Falkner-Skan solution
    '''

    nm=20
    nn=2
    L=2.0

    Uinf=1.0

    alpha=0.0

    xs=np.linspace(-L/2, L/2, nm)
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

            vels[n, 0]=Uinf*np.abs(xs[i])**alpha

            n+=1

    mu=defatm.mu
    rho0=defatm.rho

    # th11=0.664*np.sqrt(mu*np.abs(posaux[:, 0])/(rho0*Uinf))*np.ones(nm*nn)
    th11_ideal=0.66421*np.sqrt(mu/(rho0*Uinf))*np.abs(posaux[:, 0])**(1.0-alpha)/2
    th11=np.copy(th11_ideal)
    H=2.59109*np.ones_like(th11)
    Reth_ideal=th11_ideal*np.abs(vels[:, 0])*defatm.rho/defatm.mu
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

    t=tm.time()

    msh.graph_init()

    vals={
        'q':{'qx':vels[:, 0], 'qy':vels[:, 1], 'qz':vels[:, 2]}, 
        'th11':{'th11':th11},
        'H':{'H':H},
        'N':{'N':N},
        'beta':{'beta':np.zeros(nm*nn)},
        'n':{'n':nflow}
    }

    msh.set_values(vals)

    Wm=1

    weights={'Rmomx':Wm}

    value, grad=msh.calculate_graph(weights)

    print(lg.norm(value['Rmomx']), lg.norm(grad['th11']))

    immovable=['qx', 'qy', 'qz']

    stepsize=5e-3

    j=0

    for i in range(10000):
        j+=1

        if j%10==0 or j==1:
            plt.scatter(posaux[:, 0], defatm.rho*vals['q']['qx']**2*vals['th11']['th11'], label='Numeric')
            plt.plot(posaux[:, 0], th11_ideal*vals['q']['qx']**2*defatm.rho, label='Ideal')
            plt.ylim((0.0, 0.005))
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('$J_{xx}$')
            plt.legend()
            plt.show()
        
        print(sum([v@v for v, k in zip(value.values(), value) if not k in immovable]), lg.norm(grad['th11']), np.mean(vals['H']['H']))

        for n in vals:
            for p in vals[n]:
                vals[n][p]-=(stepsize/Wm if j<1000 else stepsize/(2*Wm))*grad[p]

        msh.set_values(vals)

        value, grad=msh.calculate_graph(weights)

    plt.scatter(posaux[:, 0], defatm.rho*vals['q']['qx']**2*vals['th11']['th11'], label='Numeric')
    plt.plot(posaux[:, 0], th11_ideal*vals['q']['qx']**2*defatm.rho, label='Ideal')
    plt.ylim((0.0, 0.005))
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('$J_{xx}$')
    plt.legend()
    plt.show()

test_laminar_wedge()
