from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
import scipy.interpolate as sinterp
import time as tm
from math import *

import toolkit

class network:
    #class encompassing a network of cells
    def __init__(self, mat=[[]], velmat=[[]], idisc=100, jdisc=100, ndisc=40, nstrategy=lambda x: (np.exp(x)-1.0)/(exp(1)-1.0), sstrategy=lambda x: (np.sin(pi*x-pi/2)+1)/2, \
        cstrategy=lambda x: (np.sin(pi*x-pi/2)+1)/2, delta_heuristics=lambda x, y: 1.0, thickness=1.0, rho=1.225, mu=1.72e-5):
        self.rho=rho
        self.mu=mu
        #mat: list of lists with point arrays
        #for wings or their equivalent surfaces: columns are refer to lx direction strips and rows to ly direction strips
        #delta_heuristics: a function defining initial guess for ratio (delta(x, y)/thickness kwarg), delta the maximum BL thickness considering grid geometry

        #Coordinate system for external velocity data has no relationship with local coordinate systems
        #local normal vectors are normalized in toolkit.Mtosys_gen subroutine

        #WARNING: first provided lx position (first row) must be referrent to the flow's attachment line, as provided by chocho's panel method interface functions
        
        #grid position definitions
        self.idisc=idisc
        self.jdisc=jdisc
        self.ndisc=ndisc
        self.origin=np.zeros((idisc, jdisc, 3))
        self.qes=np.zeros((idisc, jdisc, 3)) #stored in absolute coordinate system
        self.thicks=np.zeros((idisc, jdisc, ndisc))
        lx=np.linspace(0.0, 1.0, len(mat[0]))
        ly=np.linspace(0.0, 1.0, len(mat))
        xpos=np.array([[mat[i][j][0] for i in range(len(mat))] for j in range(len(mat[0]))])
        ypos=np.array([[mat[i][j][1] for i in range(len(mat))] for j in range(len(mat[0]))])
        zpos=np.array([[mat[i][j][2] for i in range(len(mat))] for j in range(len(mat[0]))])
        ues=np.array([[velmat[i][j][0] for i in range(len(mat))] for j in range(len(mat[0]))])
        ves=np.array([[velmat[i][j][1] for i in range(len(mat))] for j in range(len(mat[0]))])
        wes=np.array([[velmat[i][j][2] for i in range(len(mat))] for j in range(len(mat[0]))])
        xspline=sinterp.RectBivariateSpline(lx, ly, xpos)
        yspline=sinterp.RectBivariateSpline(lx, ly, ypos)
        zspline=sinterp.RectBivariateSpline(lx, ly, zpos)
        uespline=sinterp.RectBivariateSpline(lx, ly, ues)
        vespline=sinterp.RectBivariateSpline(lx, ly, ves)
        wespline=sinterp.RectBivariateSpline(lx, ly, wes)
        
        del xpos, ypos, zpos, ues, ves, wes
        
        #redefining in new, interpolated mesh's local coordinates
        lx=np.linspace(0.0, 1.0, idisc)
        ly=np.linspace(0.0, 1.0, jdisc)
        self.dlx=lx[1]-lx[0]
        self.dly=ly[1]-ly[0]
        xpos=xspline(lx, ly)
        ypos=yspline(lx, ly)
        zpos=zspline(lx, ly)
        self.origin[:, :, 0]=xpos
        self.origin[:, :, 1]=ypos
        self.origin[:, :, 2]=zpos
        ues=uespline(lx, ly)
        ves=vespline(lx, ly)
        wes=wespline(lx, ly)
        self.qes[:, :, 0]=ues
        self.qes[:, :, 1]=ves
        self.qes[:, :, 2]=wes

        #defining local normal vectors
        self.dxdlx=xspline(lx, ly, dx=1)
        self.dydlx=yspline(lx, ly, dx=1)
        self.dzdlx=zspline(lx, ly, dx=1)
        self.dxdly=xspline(lx, ly, dy=1)
        self.dydly=yspline(lx, ly, dy=1)
        self.dzdly=zspline(lx, ly, dy=1)
        nvects=np.zeros((idisc, jdisc, 3), dtype='double', order='F')
        nvects[:, :, 0]=self.dydlx*self.dzdly-self.dydly*self.dzdlx
        nvects[:, :, 1]=self.dzdlx*self.dxdly-self.dzdly*self.dxdlx
        nvects[:, :, 2]=self.dxdlx*self.dydly-self.dxdly*self.dydlx

        self.Mtosys, self.Mtouni=toolkit.mtosys_gen(ues, ves, wes, nvects)

        #calculating pressure gradients in local coordinate system
        press=-rho*(self.qes[:, :, 0]**2+self.qes[:, :, 1]**2+self.qes[:, :, 2]**2)/2
        pressderiv=toolkit.lambda_grad(press, self.dlx, self.dly)
        self.gradmat=toolkit.surfgradmat(self.Mtosys, self.dxdlx, self.dydlx, self.dzdlx, self.dxdly, self.dydly, self.dzdly)
        self.pressgrad=toolkit.calcgrad(self.Mtosys, pressderiv, self.gradmat, True)
        
        lthick=np.linspace(0.0, 1.0, ndisc, dtype='double')
        thickdist=nstrategy(lthick)
        for i in range(idisc):
            for j in range(jdisc):
                self.thicks[i, j, :]=delta_heuristics(lx[i], ly[j])*thickdist*thickness
        
        thickspline=sinterp.UnivariateSpline(lthick, thickdist)
        self.dlt=lthick[1]-lthick[0]
        self.dydlt=thickspline(lthick, nu=1)
        
        del thickdist, pressderiv, press, thickspline
        
        #velocity profiles initialized in local coordinate systems
        self.us=np.zeros((idisc, jdisc, ndisc), dtype='double', order='F')
        self.vs=np.zeros((idisc, jdisc, ndisc), dtype='double', order='F')
        self.ws=np.zeros((idisc, jdisc, ndisc), dtype='double', order='F')
        #lateral gradients of velocities (set as zero for initial guess)
        self.dudz=np.zeros((idisc, jdisc, ndisc))
        self.dwdz=np.zeros((idisc, jdisc, ndisc))

        self.delta=np.zeros((idisc, jdisc))
        self.attached=np.array([[True]*self.jdisc]*self.idisc)
        self.hasdelta=False

        self.setcontour()
        self.setattachment()
    def setcontour(self): #set contour conditions to external flow velocity at BL edge and no-slip, no-penetration condition at wall
        for i in range(self.idisc):
            for j in range(self.jdisc): #external velocities converted to local coordinate systems
                vec=self.Mtosys[i, j, :, :]@self.qes[i, j, :]
                self.us[i, j, -1]=vec[0]
                self.vs[i, j, -1]=vec[1]
                self.ws[i, j, -1]=vec[2]
                self.us[i, j, 0]=0.0
                self.vs[i, j, 0]=0.0
                self.ws[i, j, 0]=0.0
    def setattachment(self): #set attachment line velocities in local coordinate system
        for j in range(self.jdisc):
            self.us[0, j, 1:-1]=self.us[0, j, -1]
            self.vs[0, j, 1:-1]=self.vs[0, j, -1]
            self.ws[0, j, 1:-1]=self.ws[0, j, -1]
    def calcnorm(self, i): #returns dudy, dwdy, d2udy2, d2wdy2
        dudy=toolkit.calcnormgrad(self.us[i, :, :], self.thicks[i, :, :], self.dlt, self.dydlt)
        dwdy=toolkit.calcnormgrad(self.ws[i, :, :], self.thicks[i, :, :], self.dlt, self.dydlt)
        d2udy2=toolkit.calcnormgrad(dudy, self.thicks[i, :, :], self.dlt, self.dydlt)
        d2wdy2=toolkit.calcnormgrad(dwdy, self.thicks[i, :, :], self.dlt, self.dydlt)
        return dudy, dwdy, d2udy2, d2wdy2
    def propagate(self, i): #propagates a velocity profile from row i to row i+1 invoking BL equations to compute velocity gradients
        #defining y components of velocity gradients
        dudy, dwdy, d2udy2, d2wdy2=self.calcnorm(i)
        #z components have already been computed (or guessed, if first iteration)
        #now, deducing v velocity along vel. profile:
        self.vs[i, :, 1:-1]=toolkit.calcv(self.dudz[i, :, :], self.dwdz[i, :, :], self.us[i, :, :], self.ws[i, :, :], \
            self.pressgrad[i, :, 0], self.mu, self.rho, d2udy2, self.thicks[i, :, :])
        dudx, dwdx=toolkit.blexercise(self.mu, self.rho, self.us[i, :, :], self.vs[i, :, :], self.ws[i, :, :], \
            self.dudz[i, :, :], self.dwdz[i, :, :], dudy, dwdy, self.pressgrad[i, :, :], \
                d2udy2, d2wdy2)
        self.us[i+1, :, 1:-1], self.ws[i+1, :, 1:-1]=toolkit.blpropagate(self.origin[i, :, :], self.origin[i+1, :, :], \
            self.thicks[i, :, :], self.thicks[i+1, :, :], self.us[i, :, :], self.ws[i, :, :], self.Mtosys[i, :, :, :], self.Mtosys[i+1, :, :, :], \
                self.Mtouni[i, :, :, :], self.Mtouni[i+1, :, :, :], dudx, dwdx, dudy[:, 1:-1], \
                    dwdy[:, 1:-1], self.dudz[i, :, 1:-1], self.dwdz[i, :, 1:-1])
        self.check_detachment(i+1)
        self.hasdelta=False
    def check_detachment(self, i): #function to check whether flow detatchment occurs at a given position based on Goldstein's singularity
        self.attached[i, :]=toolkit.checkattach(self.us[i, :, :])
    def calc_delta(self): #calculate delta based on value for local streamwise velocity
        prop=np.zeros((self.idisc, self.jdisc, self.ndisc))
        for i in range(self.idisc):
            for j in range(self.jdisc):
                prop[i, j, :]=1.0-self.us[i, j, :]/self.us[i, j, -1]
        self.delta=toolkit.intthick(prop, self.thicks, self.dydlt, self.dlt)
        self.hasdelta=True
    def plot_surfgeom(self, color='blue', ax=None, fig=None, show=True, xlim=[], ylim=[], zlim=[]): #plot surface geometry as matplotlib surface
        if fig==None:
            fig=plt.figure()
        if ax==None:
            ax=plt.axes(projection='3d')
        ax.plot_surface(self.origin[:, :, 0], self.origin[:, :, 1], self.origin[:, :, 2], color=color)
        if len(xlim)!=0:
            ax.set_xlim3d(xlim[0], xlim[1])
        if len(ylim)!=0:
            ax.set_ylim3d(ylim[0], ylim[1])
        if len(zlim)!=0:
            ax.set_zlim3d(zlim[0], zlim[1])
        if show:
            plt.show()
    def plot_delta(self, color='gray', ax=None, fig=None, show=True, xlim=[], ylim=[], zlim=[], factor=1.0): #plot local boundary layer thickness
        if not self.hasdelta:
            self.calc_delta()
        if fig==None:
            fig=plt.figure()
        if ax==None:
            ax=plt.axes(projection='3d')
        self.plot_surfgeom(color='blue', ax=ax, fig=fig, show=False)
        thickpos=np.zeros((self.idisc, self.jdisc, 3))
        for i in range(self.idisc):
            for j in range(self.jdisc):
                thickpos[i, j, :]=self.origin[i, j, :]+self.Mtosys[i, j, 1, :]*self.delta[i, j]*factor
        print(thickpos)
        ax.plot_wireframe(thickpos[:, :, 0], thickpos[:, :, 1], thickpos[:, :, 2], color=color)
        if len(xlim)!=0:
            ax.set_xlim3d(xlim[0], xlim[1])
        if len(ylim)!=0:
            ax.set_ylim3d(ylim[0], ylim[1])
        if len(zlim)!=0:
            ax.set_zlim3d(zlim[0], zlim[1])
        if show:
            plt.show()
        
        

#test case for constructors: flat plate
ps=[[np.array([x, y, 0.0]) for x in np.linspace(0.0, 1.0, 100)] for y in np.linspace(0.0, 1.0, 50)]
qes=[[np.array([1.0, 0.0, 0.0]) for u in np.linspace(0.0, 1.0, 100)] for v in np.linspace(0.0, 1.0, 50)]
t=tm.time()
ntw=network(mat=ps, velmat=qes, idisc=500, jdisc=50, ndisc=50, thickness=0.06, delta_heuristics=lambda x, y : (x+1.0)/2)#, nstrategy=lambda x: x)
for i in range(ntw.idisc-1):
    ntw.propagate(i)
ntw.calc_delta()
print(tm.time()-t, ' s')
ntw.plot_delta(factor=100.0)
'''plt.plot(ntw.origin[ntw.attached[:, 0], 0, 0], ntw.delta[ntw.attached[:, 0], 0])
plt.show()
for i in range(0, ntw.idisc, 20):
    plt.plot(ntw.us[i, 0, :], ntw.thicks[i, 0, :])
    plt.show()'''