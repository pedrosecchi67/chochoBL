import numpy as np
import numpy.linalg as lg
import time as tm

import abaqus as abq
from closure import *
import station as stat

def FS_extrap_wall(s, nu, Uinf): #returns Falkner-Skan boundary layer thickness according to a normal-to-wall inviscid solution (wedge angle=pi rad)
    return np.sqrt(s*nu/Uinf)

class mesh:
    def __init__(self, atm_props=stat.defatm, clsr=stat.defclsr, posits=np.zeros((2, 2, 3)), vels=np.zeros((2, 2, 3)), Uinf=1.0, extrapfun=FS_extrap_wall, gamma=1.4):
        self.Uinf=Uinf
        self.vels=vels
        self.posits=posits
        self.atm_props=atm_props
        self.clsr=clsr
        self.nm=np.size(self.vels, 0)
        self.nn=np.size(self.vels, 1)
        self.dlx=1.0/self.nm
        self.dly=1.0/self.nn
        self.local_coordinate_define()
        self.matrix=[[None]*self.nm]*self.nn #dimensions are inversed in order : columns (x variation) are identified by the first index.
        #for faster access only
        self.attachinds=self.identify_starting_points()
        self.contour_apply()
        self.extrap=lambda s: extrapfun(s, self.atm_props.mu/(self.atm_props.rho*(1.0+(gamma-1.0)*(Uinf/self.atm_props.v_sonic)**2/2)**(gamma/(gamma-1.0))), Uinf)
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
        self.Hu=self.calc_Hessian(self.parvels[:, :, 0])
        self.Hv=self.calc_Hessian(self.parvels[:, :, 1])
        self.Hw=self.calc_Hessian(self.parvels[:, :, 2])
        self.velderivs_define()
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
    def calc_Hessian(self, props):
        H=np.zeros((3, 3, self.nm, self.nn))
        dx, dy, dz=self.calc_derivative(props)
        H[0, 0, :, :], H[0, 1, :, :], H[0, 2, :, :]=self.calc_derivative(dx)
        H[1, 0, :, :], H[1, 1, :, :], H[1, 2, :, :]=self.calc_derivative(dy)
        H[2, 0, :, :], H[2, 1, :, :], H[2, 2, :, :]=self.calc_derivative(dz)
        return H
    def velderivs_define(self):
        dw_ds=self.calc_velderivs(vel='w', direct='s')
        dq_ds=self.calc_velderivs(vel='q', direct='s')

        Me=self.qes/self.atm_props.v_sonic

        dq_dx, dq_dy, dq_dz=self.calc_derivative(self.qes)
        self.dq_dx=dq_dx*self.s[:, :, 0]+dq_dy*self.s[:, :, 1]+dq_dz*self.s[:, :, 2]
        self.dq_dz=dq_dx*self.c[:, :, 0]+dq_dy*self.c[:, :, 1]+dq_dz*self.c[:, :, 2]

        d2q_dsdx, d2q_dsdy, d2q_dsdz=self.calc_derivative(self.dq_dx)
        self.d2q_dx2=d2q_dsdx*self.s[:, :, 0]+d2q_dsdy*self.s[:, :, 1]+d2q_dsdz*self.s[:, :, 2]
        self.d2q_dxdz=d2q_dsdx*self.c[:, :, 0]+d2q_dsdy*self.c[:, :, 1]+d2q_dsdz*self.c[:, :, 2]

        self.betas=np.arctan2(dw_ds, (1.0-Me**2)*dq_ds)
        dbeta_dx, dbeta_dy, dbeta_dz=self.calc_derivative(self.betas)
        self.dbetas_dx=dbeta_dx*self.s[:, :, 0]+dbeta_dy*self.s[:, :, 1]+dbeta_dz*self.s[:, :, 2]
        self.dbetas_dz=dbeta_dx*self.c[:, :, 0]+dbeta_dy*self.c[:, :, 1]+dbeta_dz*self.c[:, :, 2]
    def identify_starting_points(self):
        startinds=np.zeros(self.nn, dtype='int')
        for i in range(self.nn):
            startinds[i]=np.argmin(self.vels[:, i, 0])
        return startinds
    def contour_apply(self):
        for i, ind in zip(range(self.nn), self.attachinds):
            self.matrix[i][ind]=stat.station(delta=0.0, qe=self.qes[ind, i], dq_dx=self.dq_dx[ind, i], dq_dz=self.dq_dz[ind, i], d2q_dx2=self.d2q_dx2[ind, i], d2q_dxdz=self.d2q_dxdz[ind, i], \
                props=self.atm_props, clsr=self.clsr, Uinf=self.Uinf)
    def LE_extrapolate(self):
        for i, ind in zip(range(self.nn), self.attachinds):
            if ind!=self.nm:
                self.matrix[i][ind+1]=stat.station(delta=self.extrap(self.s[ind+1, i, :]@(self.posits[ind+1, i, :]-self.posits[ind, i, :])), qe=self.qes[ind+1, i], dq_dx=self.dq_dx[ind+1, i], \
                    dq_dz=self.dq_dz[ind+1, i], d2q_dx2=self.d2q_dx2[ind+1, i], d2q_dxdz=self.d2q_dxdz[ind+1, i], props=self.atm_props, clsr=self.clsr, Uinf=self.Uinf)
            if ind!=0:
                self.matrix[i][ind-1]=stat.station(delta=self.extrap(self.s[ind-1, i, :]@(self.posits[ind-1, i, :]-self.posits[ind, i, :])), qe=self.qes[ind-1, i], dq_dx=self.dq_dx[ind-1, i], \
                    dq_dz=self.dq_dz[ind-1, i], d2q_dx2=self.d2q_dx2[ind-1, i], d2q_dxdz=self.d2q_dxdz[ind-1, i], props=self.atm_props, clsr=self.clsr, Uinf=self.Uinf)
    def propagate(self):
        for i in range(self.nn):
            self.matrix[i][self.attachinds[i]].calc_data()
            for j in range(self.attachinds[i]+2, self.nm):
                ds, dc=self.matrix[i][j-1].calcpropag()
                relpos=self.posits[j, i, :]-self.posits[j-1, i, :]
                d=self.matrix[i][j-1].delta+ds*relpos@self.s[j-1, i, :]+dc*relpos@self.c[j-1, i, :]
                self.matrix[i][j]=stat.station(delta=d, qe=self.qes[j, i], dq_dx=self.dq_dx[j, i], dq_dz=self.dq_dz[j, i], d2q_dx2=self.d2q_dx2[j, i], d2q_dxdz=self.d2q_dxdz[j, i], clsr=self.clsr, \
                    props=self.atm_props, Uinf=self.Uinf)
            for j in range(self.attachinds[i]-2, -1, -1):
                ds, dc=self.matrix[i][j+1].calcpropag()
                relpos=self.posits[j, i, :]-self.posits[j+1, i, :]
                d=self.matrix[i][j+1].delta+ds*relpos@self.s[j+1, i, :]+dc*relpos@self.c[j+1, i, :]
                self.matrix[i][j]=stat.station(delta=d, qe=self.qes[j, i], dq_dx=self.dq_dx[j, i], dq_dz=self.dq_dz[j, i], d2q_dx2=self.d2q_dx2[j, i], d2q_dxdz=self.d2q_dxdz[j, i], clsr=self.clsr, \
                    props=self.atm_props, Uinf=self.Uinf)
            self.matrix[i][-1].calc_data()
            self.matrix[i][0].calc_data()
    def calculate(self):
        self.LE_extrapolate()
        self.propagate()

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
msh.calculate()
print(tm.time()-t)