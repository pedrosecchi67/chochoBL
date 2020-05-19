import chochoBL as cho

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def test_NACA0012():
    #loading data
    velsx=np.loadtxt('N0012AR10_5deg_vels0.dat')
    velsy=np.loadtxt('N0012AR10_5deg_vels1.dat')
    velsz=np.loadtxt('N0012AR10_5deg_vels2.dat')
    positsx=np.loadtxt('N0012AR10_5deg_posits0.dat')
    positsy=np.loadtxt('N0012AR10_5deg_posits1.dat')
    positsz=np.loadtxt('N0012AR10_5deg_posits2.dat')

    #reshaping
    xdisc=np.size(velsx, 0)
    ydisc=np.size(velsx, 1)

    vels=np.zeros((xdisc, ydisc, 3))
    posits=np.zeros((xdisc, ydisc, 3))

    vels[:, :, 0]=velsx
    vels[:, :, 1]=velsy
    vels[:, :, 2]=velsz
    posits[:, :, 0]=positsx
    posits[:, :, 1]=positsy
    posits[:, :, 2]=positsz

    del velsx, velsy, velsz, positsx, positsy, positsz

    msh=cho.mesh(posits=posits, vels=vels, Uinf=100.0)

    msh.calculate()

    #fig=plt.figure()
    #ax=plt.axes(projection='3d')
    
    ds=np.zeros((len(msh.matrix), len(msh.matrix[0])))
    Lambdas=np.zeros((len(msh.matrix), len(msh.matrix[0])))
    print(msh.attachinds)
    for i, strip in enumerate(msh.matrix):
        for j, elem in enumerate(strip):
            ds[i, j]=elem.delta
            Lambdas[i, j]=elem.th[0, 0]
    
    plt.plot(posits[38:-3, 20, 0], ds[20, 38:-3])
    plt.ylim((0.0, 0.008))
    plt.show()
    plt.plot(posits[38:-3, 20, 0], Lambdas[20, 38:-3])
    plt.ylim((0.0, 0.003))
    plt.show()

    #ax.plot_surface()

test_NACA0012()