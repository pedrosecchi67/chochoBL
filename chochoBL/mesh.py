import numpy as np
import numpy.linalg as lg
import time as tm

import abaqus as abq
import closure as clsr
import station as stat

class mesh:
    def __init__(self, atm_props=stat.defatm, clsr=stat.defclsr, posits=np.zeros((2, 2, 3)), vels=np.zeros((2, 2, 3))):
        self.vels=vels
        self.posits=posits
        self.atm_props=atm_props
        self.clsr=clsr
        self.nm=np.size(self.vels, 0)
        self.nn=np.size(self.vels, 1)
        self.dlx=1.0/self.nm
        self.dly=1.0/self.nn
        self.local_coordinate_define()
    def local_coordinate_define(self):
        self.dxdlx, self.dxdly=self.calc_derivative_aux(self.posits[:, :, 0])
        self.dydlx, self.dydly=self.calc_derivative_aux(self.posits[:, :, 1])
        self.dzdlx, self.dzdly=self.calc_derivative_aux(self.posits[:, :, 2])
        self.n=np.zeros((self.nm, self.nn, 3))
        self.n[:, :, 0]=self.dydlx*self.dzdly-self.dzdlx*self.dydly
        self.n[:, :, 1]=self.dzdlx*self.dxdly-self.dxdlx*self.dzdly
        self.n[:, :, 2]=self.dxdlx*self.dydly-self.dydlx*self.dxdly
        norms=np.sqrt(self.n[:, :, 0]**2+self.n[:, :, 1]**2+self.n[:, :, 2]**2)
        self.n[:, :, 0]/=norms
        self.n[:, :, 1]/=norms
        self.n[:, :, 2]/=norms
        self.parvels=np.copy(self.vels)
        projs=(self.vels[:, :, 0]*self.n[:, :, 0]+self.vels[:, :, 1]*self.n[:, :, 1]+self.vels[:, :, 2]*self.n[:, :, 2])
        self.parvels[:, :, 0]-=self.n[:, :, 0]*projs
        self.parvels[:, :, 1]-=self.n[:, :, 1]*projs
        self.parvels[:, :, 2]-=self.n[:, :, 2]*projs
        self.qes=np.sqrt(self.parvels[:, :, 0]**2+self.parvels[:, :, 1]**2+self.parvels[:, :, 2]**2)
        self.s=np.zeros((self.nm, self.nn, 3))
        self.s[:, :, 0]=self.parvels[:, :, 0]/self.qes
        self.s[:, :, 1]=self.parvels[:, :, 1]/self.qes
        self.s[:, :, 2]=self.parvels[:, :, 2]/self.qes
        self.c=np.zeros((self.nm, self.nn, 3))
        self.c[:, :, 0]=self.n[:, :, 1]*self.s[:, :, 2]-self.n[:, :, 2]*self.s[:, :, 1]
        self.c[:, :, 1]=self.n[:, :, 2]*self.s[:, :, 0]-self.n[:, :, 0]*self.s[:, :, 2]
        self.c[:, :, 2]=self.n[:, :, 0]*self.s[:, :, 1]-self.n[:, :, 1]*self.s[:, :, 0]
        #defining gradient matrix
        self.gradient=np.zeros((3, 2, self.nm, self.nn))
        dets=self.dxdlx*self.dydly*self.n[:, :, 2]+self.dydlx*self.dzdly*self.n[:, :, 0]+self.dzdlx*self.dxdly*self.n[:, :, 1]\
            -self.n[:, :, 0]*self.dydly*self.dzdlx-self.n[:, :, 1]*self.dzdly*self.dxdlx-self.n[:, :, 2]*self.dxdly*self.dydlx
        self.gradient[0, 0, :, :]=(self.dydly*self.n[:, :, 2]-self.n[:, :, 1]*self.dzdly)/dets
        self.gradient[0, 1, :, :]=(self.n[:, :, 1]*self.dzdlx-self.dydlx*self.n[:, :, 2])/dets
        self.gradient[1, 0, :, :]=(self.n[:, :, 0]*self.dzdly-self.n[:, :, 2]*self.dxdly)/dets
        self.gradient[1, 1, :, :]=(self.dxdlx*self.n[:, :, 2]-self.n[:, :, 0]*self.dzdlx)/dets
        self.gradient[2, 0, :, :]=(self.dxdly*self.n[:, :, 1]-self.n[:, :, 0]*self.dydly)/dets
        self.gradient[2, 1, :, :]=(self.n[:, :, 0]*self.dydlx-self.dxdlx*self.n[:, :, 1])/dets
        self.J=np.zeros((3, 3, self.nm, self.nn)) #tensor S=[dudx, dudy, dudz; dv...]
        for i in range(3):
            self.J[i, 0, :, :], self.J[i, 1, :, :], self.J[i, 2, :, :]=self.calc_derivative(self.parvels[:, :, i])
    def calc_derivative_aux(self, data):
        dvslx=np.zeros((self.nm, self.nn))
        dvsly=np.zeros((self.nm, self.nn))
        mids=(data[1:, :]-data[:-1, :])/self.dlx
        dvslx[:-1, :]+=mids
        dvslx[1:, :]+=mids
        dvslx[1:-1, :]/=2
        mids=(data[:, 1:]-data[:, :-1])/self.dly
        dvsly[:, :-1]+=mids
        dvsly[:, 1:]+=mids
        dvsly[:, 1:-1]/=2
        return dvslx, dvsly
    def calc_derivative(self, data):
        dvslx, dvsly=self.calc_derivative_aux(data)
        return self.gradient[0, 0, :, :]*dvslx+self.gradient[0, 1, :, :]*dvsly, self.gradient[1, 0, :, :]*dvslx+self.gradient[1, 1, :, :]*dvsly, \
            self.gradient[2, 0, :, :]*dvslx+self.gradient[2, 1, :, :]*dvsly
    def calc_velderivs(self, vel='q', direct='s'):
        if vel=='q':
            veldirs=self.s
        elif vel=='w':
            veldirs=self.c
        if direct=='s':
            direction=self.s
        elif direct=='c':
            direction=self.c
        trans=np.zeros((3, self.nm, self.nn))
        for i in range(3):
            trans[i, :, :]=self.J[i, 0, :, :]*direction[:, :, 0]+self.J[i, 1, :, :]*direction[:, :, 1]+self.J[i, 2, :, :]*direction[:, :, 2]
        return veldirs[:, :, 0]*trans[0, :, :]+veldirs[:, :, 1]*trans[1, :, :]+veldirs[:, :, 2]*trans[2, :, :]

nm=100
nn=50
xs=np.linspace(0.0, 1.0, nm)
ys=np.linspace(0.0, 1.0, nn)
posits=np.zeros((nm, nn, 3))
for i in range(nm):
    for j in range(nn):
        posits[i, j, 0]=xs[i]
        posits[i, j, 1]=ys[j]
vels=np.zeros((nm, nn, 3))
vels[:, :, 0]=1.0
t=tm.time()
msh=mesh(posits=posits, vels=vels)
print(tm.time()-t)