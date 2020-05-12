import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.interpolate as sinterp
import time as tm
import fluids.atmosphere as atm
import cloudpickle as pck
import os

import abaqus as abq

Gersten_Herwig_A=6.1e-4
Gersten_Herwig_B=1.43e-3
Gersten_Herwig_Lambda=(Gersten_Herwig_A+Gersten_Herwig_B)**(1.0/3)
Von_Karman_kappa=0.41
def Gersten_Herwig_LOTW(yp):
    return (np.log((Gersten_Herwig_Lambda*yp+1.0)/np.sqrt((Gersten_Herwig_Lambda*yp)**2-\
        Gersten_Herwig_Lambda*yp+1.0))/3+(np.arctan((2*Gersten_Herwig_Lambda*yp-1.0)/np.sqrt(3))+np.pi/6)/np.sqrt(3))/Gersten_Herwig_Lambda+\
            np.log(1.0+Von_Karman_kappa*Gersten_Herwig_B*(yp)**4)/(4*Von_Karman_kappa)

def laminar_profile_analyse(Lambda, Red, a, b, M, disc=100, strategy=lambda x: x):
    '''
    Analyse laminar profile, according to function describing streamwise profile (fun) and
    crossflow profile (M, uc/q=M(eta)*fun(eta)*tan(crossflow))
    ======
    parameters
    ======
    * a, b: streamwise profile functions: f(eta)=a(eta)+b(eta)*Lambda, Lambda equal to fpp at the wall
    * M: crossflow profile (M, uc/q=M(eta)*fun(eta)*tan(crossflow))
    * disc: discretization for integration along the boundary layer
    * strategy: function distributing etas across BL (etas=strategy(np.linspace(0.0, 1.0, disc)))
    ======
    returns
    ======
    deltastar_x/delta, deltastar_z/delta, thxx/delta, thxz/(delta*tan(crossflow)), thzx/(delta*tan(crossflow)), thzz/(delta*tan(crossflow)**2), Cf
    '''

    fun=lambda eta: a(eta)+Lambda*b(eta)

    etas=strategy(np.linspace(0.0, 1.0, disc))
    fs=fun(etas)
    ms=M(etas)

    f=np.trapz(fs, x=etas)
    f2=np.trapz(fs**2, x=etas)
    fm=np.trapz(fs*ms, x=etas)
    f2m=np.trapz(fs**2*ms, x=etas)
    f2m2=np.trapz((fs*ms)**2, x=etas)

    dx=1.0-f
    dz=fm
    thxx=f-f2
    thxz=fm-f2m
    thzx=f2m
    thzz=f2m2

    h=1.0/disc
    fp=fun(h)/h
    Cf=fp*2/Red #Cf*Reynolds number in respect to delta

    return dx, dz, thxx, thxz, thzx, thzz, Cf

def turbulent_profile_analyse(A, deltastar, M=lambda x: 1.0-x**2, LOTW=Gersten_Herwig_LOTW, outter_profile=lambda x: np.sin(np.pi*x/2)**2, disc=100, strategy=lambda x: x):
    '''
    Analyse turbulent profile, according to function describing Law of The Wall (default is Gersten-Herwig's), outter profile and
    crossflow profile (M, uc/q=M(eta)*f(eta)*tan(crossflow))
    ======
    parameters
    ======
    * LOTW: Law of The Wall
    * outter_profile: outter profile shape function. up(yp)=LOTW(yp)+A*w(yp/deltap)
    * M: crossflow profile (M, uc/q=M(eta)*f(eta)*tan(crossflow))
    * disc: discretization for integration along the boundary layer
    * strategy: function distributing etas across BL (etas=strategy(np.linspace(0.0, 1.0, disc)))
    ======
    returns
    ======
    deltastar_x/delta, deltastar_z/delta, thxx/delta, thxz/(delta*tan(crossflow)), thzx/(delta*tan(crossflow)), thzz/(delta*tan(crossflow)**2), Cf
    '''
    etas=strategy(np.linspace(0.0, 1.0, disc))
    yps=etas*deltastar

    ups=LOTW(yps)+A*outter_profile(etas)
    
    h=1.0/disc
    wpp=(outter_profile(h)-2*outter_profile(0.0)+outter_profile(-h))/h**2

    Ut=1.0/ups[-1]
    Cf=2*Ut**2

    fs=ups*Ut
    ms=M(etas)

    f=np.trapz(fs, x=etas)
    f2=np.trapz(fs**2, x=etas)
    fm=np.trapz(fs*ms, x=etas)
    f2m=np.trapz(fs**2*ms, x=etas)
    f2m2=np.trapz((fs*ms)**2, x=etas)

    dx=1.0-f
    dz=fm
    thxx=f-f2
    thxz=fm-f2m
    thzx=f2m
    thzz=f2m2

    Lambda=A*Ut*wpp

    return dx, dz, thxx, thxz, thzx, thzz, Cf

class closure:
    def __init__(self, disc):
        self.disc=disc

    def dump(self, fname, ext_append=True):
        '''
        method to dump a set of closure relationships to a binary file.
        ======
        args
        ======
        fname: file name (or complete directory) within which to save the pickled closure relationships
        ext_append: wether to add .cls to file name
        '''

        fil=open(fname+'.cls' if ext_append else '', 'wb')
        pck.dump(self, fil)
        fil.close()
    
    def build(self, Lambdas, Reds, thxx, thxz, thzx, thzz, dx, dz, Cf, log_Reynolds=False):
        '''
        Build closure relationship as a spline mapping of provided properties
        '''

        self.thxx=abq.abaqus(Lambdas, np.log(Reds) if log_Reynolds else Reds, thxx)
        self.thxz=abq.abaqus(Lambdas, np.log(Reds) if log_Reynolds else Reds, thxz)
        self.thzx=abq.abaqus(Lambdas, np.log(Reds) if log_Reynolds else Reds, thzx)
        self.thzz=abq.abaqus(Lambdas, np.log(Reds) if log_Reynolds else Reds, thzz)
        self.dx=abq.abaqus(Lambdas, np.log(Reds) if log_Reynolds else Reds, dx)
        self.dz=abq.abaqus(Lambdas, np.log(Reds) if log_Reynolds else Reds, dz)
        self.Cf=abq.abaqus(Lambdas, np.log(Reds) if log_Reynolds else Reds, Cf)

        self.log_Reynolds=log_Reynolds

    def __call__(self, Lambda, Red, nu=False):
        '''
        return values from closure relationships
        ======
        args
        ======
        Lambda, Red: x and y values to look for in splines. Lambda is fpp value at the wall, or -delta**2*dq_dx/mu. Red is Reynolds number in
        respect to delta
        nu: boolean on whether to differentiate return values. If True, returns theta tensor ((thxx, thxz/tan(crossflow)), (thzx/tan(crossflow), thzz/tan2(crossflow))), 
        or th, returned as derivated values dth/dLambda and dth/dRed.
        else returns th, deltastar_x, deltastar_z/tan(crossflow), Cf (not differentiated)
        '''
        
        rd=np.log(Red) if self.log_Reynolds else Red

        if nu:
            #return theta tensor Lambda and Red derivatives
            return np.array([[self.thxx(Lambda, rd, dx=1), self.thxz(Lambda, rd, dx=1)], [self.thzx(Lambda, rd, dx=1), self.thzz(Lambda, rd, dx=1)]]), \
                np.array([[self.thxx(Lambda, rd, dy=1), self.thxz(Lambda, rd, dy=1)], [self.thzx(Lambda, rd, dy=1), self.thzz(Lambda, rd, dy=1)]])/(Red if self.log_Reynolds else 1.0)
        
        else:
            #return tensor, dx, dz, Cf
            return np.array([[self.thxx(Lambda, rd), self.thxz(Lambda, rd)], [self.thzx(Lambda, rd), self.thzz(Lambda, rd)]]), \
                self.dx(Lambda, rd), self.dz(Lambda, rd), self.Cf(Lambda, rd)

class laminar_closure(closure):
    def __init__(self, disc=100, a=lambda x: 2*x-2*x**3+x**4, b=lambda x: -x*(1.0-x)**3/6, M=lambda eta: 1.0-eta**2):
        super().__init__(disc)

        #recording velocity profiles: f(eta)=a(eta)+Lambda*b(eta), Lambda=fpp at the wall
        self.a=a
        self.b=b
        self.M=M
    
    def build(self, Lambdas=np.linspace(-2.0, 4.9, 50), Reds=10**np.linspace(2.0, 6.0, 50), \
        strategy=lambda x: x, log_Reynolds=True):
        '''
        Build laminar closure relationships for laminar flow for velocity profile f(eta)=a(eta)+Lambda*b(eta), 
        Lambda=fpp at the wall (deduced from freestream). log_Reynolds sets input Reynolds numbers to logarythmic scale before splining.
        '''

        nm=len(Lambdas)
        nn=len(Reds)

        dx=np.zeros((nm, nn))
        dz=np.zeros((nm, nn))
        thxx=np.zeros((nm, nn))
        thxz=np.zeros((nm, nn))
        thzx=np.zeros((nm, nn))
        thzz=np.zeros((nm, nn))
        Cf=np.zeros((nm, nn))

        for i, l in enumerate(Lambdas):
            for j, rd in enumerate(Reds):
                dx[i, j], dz[i, j], thxx[i, j], thxz[i, j], thzx[i, j], thzz[i, j], Cf[i, j]=laminar_profile_analyse(l, rd, \
                    a=self.a, b=self.b, M=self.M, disc=self.disc, strategy=strategy)
        
        super().build(Lambdas=Lambdas, Reds=Reds, thxx=thxx, thxz=thxz, thzx=thzx, thzz=thzz, dx=dx, dz=dz, Cf=Cf, log_Reynolds=log_Reynolds)

def get_A_dst(Lambda, wpp_w, Red, LOTW, initguess):
    Ut=sopt.fsolve(lambda Ut: (1.0-Lambda/wpp_w)-Ut*LOTW(Red*Ut), x0=initguess)[0]
    A=Lambda/(Ut*wpp_w)
    return A, Ut*Red

class turbulent_closure(closure):
    def __init__(self, M=lambda eta: 1.0-eta**2, LOTW=Gersten_Herwig_LOTW, w=lambda x: np.sin(np.pi*x/2)**2, disc=100):
        h=1.0/disc
        self.wpp_w=(w(h)-2*w(0)+w(-h))/h**2
        self.LOTW=LOTW
        self.w=w
        self.M=M

        super().__init__(disc)

    def build(self, Lambdas=np.linspace(-2.0, 4.9, 50), Reds=10**np.linspace(2.0, 6.0, 50), \
        strategy=lambda x: x, Ut_initguess=0.1, log_Reynolds=True):
        '''
        Build the closure relationships for a given range of Lambdas and thickness Reynolds numbers. strategy and disc arguments
        are referrent to function turbulent_profile_analyse(). log_Reynolds sets input Reynolds numbers to logarythmic scale before splining.
        '''

        nm=len(Lambdas)
        nn=len(Reds)

        dx=np.zeros((nm, nn))
        dz=np.zeros((nm, nn))
        thxx=np.zeros((nm, nn))
        thxz=np.zeros((nm, nn))
        thzx=np.zeros((nm, nn))
        thzz=np.zeros((nm, nn))
        Cf=np.zeros((nm, nn))

        for i, l in enumerate(Lambdas):
            for j, rd in enumerate(Reds):
                a, d=get_A_dst(Lambda=l, wpp_w=self.wpp_w, Red=rd, LOTW=self.LOTW, initguess=Ut_initguess)
                dx[i, j], dz[i, j], thxx[i, j], thxz[i, j], thzx[i, j], thzz[i, j], Cf[i, j]=turbulent_profile_analyse(a, d, \
                    M=self.M, LOTW=self.LOTW, outter_profile=self.w, disc=self.disc, strategy=strategy)
        
        super().build(Lambdas=Lambdas, Reds=Reds, thxx=thxx, thxz=thxz, thzx=thzx, thzz=thzz, dx=dx, dz=dz, Cf=Cf, log_Reynolds=log_Reynolds)



def read_closure(fname, ext_append=True):
    '''
    function to read a set of closure relationships from a binary file
    ======
    args
    ======
    fname: file name (or complete directory) within which to save the pickled closure relationships
    ext_append: whether to expect the name+.cls
    '''

    fil=open(fname+'.cls' if ext_append else '', 'rb')
    clsr=pck.load(fil)
    fil.close()

    return clsr

#look for default closure relationships in package folder
ordir=os.getcwd()

os.chdir(os.path.dirname(__file__))

if os.path.exists('lam_defclosure.cls'): #change and False later
    lam_defclosure=read_closure('lam_defclosure')

else:
    lam_defclosure=laminar_closure()
    lam_defclosure.build()
    lam_defclosure.dump('lam_defclosure')

if os.path.exists('turb_defclosure.cls'): #change and False later
    turb_defclosure=read_closure('turb_defclosure')

else:
    turb_defclosure=turbulent_closure()
    turb_defclosure.build()
    turb_defclosure.dump('turb_defclosure')
    
os.chdir(ordir)
del ordir