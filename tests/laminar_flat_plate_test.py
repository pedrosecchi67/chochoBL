from chochoBL import *

import numpy as np
import pytest

def test_laminar_flat_plate():
    nm=30
    nn=2
    L=1.0
    Uinf=1.0
    xs=np.sin(np.pi*np.linspace(0.0, 1.0, nm)/2)**2*L
    ys=np.linspace(0.0, 1.0, nn)
    posits=np.zeros((nm, nn, 3))
    for i in range(nm):
        for j in range(nn):
            posits[i, j, 0]=xs[i]
            posits[i, j, 1]=ys[j]
    vels=np.zeros((nm, nn, 3))
    vels[:, :, 0]=Uinf
    #t=tm.time()
    msh=mesh(posits=posits, vels=vels)
    msh.calculate()
    #print(tm.time()-t)
    ds=np.array([[elem.th[0, 0] for elem in strip] for strip in msh.matrix])

    #for i in range(len(msh.matrix[0])):
    #    print([msh.matrix[j][i].has_transition() for j in range(len(msh.matrix))])
    #    print([msh.matrix[j][i].transition for j in range(len(msh.matrix))])

    dif=abs(ds[-1, -1]-0.665*np.sqrt(defatm.mu*L/(Uinf*defatm.rho)))

    assert dif/ds[-1, -1]<1e-1, 'Wrong theta measured at edge on comparison to Blausius solution, Re %e, perc. error %f\%' % ((Uinf*L*defatm.rho)/defatm.mu, 100*dif/ds[-1, -1])

#xxs, yys=np.meshgrid(xs, ys)
#fig=plt.figure()
#ax=plt.axes(projection='3d')
#ax.plot_surface(xxs, yys, ds)
#plt.show()