import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.interpolate as sinterp
import time as tm
import abaqus as abq
import fluids.atmosphere as atm

Gersten_Herwig_A=6.1e-4
Gersten_Herwig_B=1.43e-3
Gersten_Herwig_Lambda=(Gersten_Herwig_A+Gersten_Herwig_B)**(1.0/3)
Von_Karman_kappa=0.41
def Gersten_Herwig_LOTW(yp):
    return (np.log((Gersten_Herwig_Lambda*yp+1.0)/np.sqrt((Gersten_Herwig_Lambda*yp)**2-\
        Gersten_Herwig_Lambda*yp+1.0))/3+(np.arctan((2*Gersten_Herwig_Lambda*yp-1.0)/np.sqrt(3))+np.pi/6)/np.sqrt(3))/Gersten_Herwig_Lambda+\
            np.log(1.0+Von_Karman_kappa*Gersten_Herwig_B*(yp)**4)/(4*Von_Karman_kappa)

def laminar_profile_analyse(fun, M, disc=100, strategy=lambda x: x):
    '''
    Analyse laminar profile, according to function describing streamwise profile (fun) and
    crossflow profile (M, uc/q=M(eta)*fun(eta)*tan(crossflow))
    ======
    parameters
    ======
    * fun: streamwise profile function
    * M: crossflow profile (M, uc/q=M(eta)*fun(eta)*tan(crossflow))
    * disc: discretization for integration along the boundary layer
    * strategy: function distributing etas across BL (etas=strategy(np.linspace(0.0, 1.0, disc)))
    ======
    returns
    ======
    deltastar_x/delta, deltastar_z/delta, thxx/delta, thxz/(delta*tan(crossflow)), thzx/(delta*tan(crossflow)), thzz/(delta*tan(crossflow)**2)
    '''

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

    return dx, dz, thxx, thxz, thzx, thzz

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

def get_A_dst(Lambda, wpp_w, Red, LOTW, initguess):
    Ut=sopt.fsolve(lambda Ut: (1.0-Lambda/wpp_w)-Ut*LOTW(Red*Ut), x0=initguess)[0]
    A=Lambda/(Ut*wpp_w)
    return A, Ut*Red

class closure:
    def __init__(self, M=lambda eta: 1.0-eta**2, LOTW=Gersten_Herwig_LOTW, w=lambda x: np.sin(np.pi*x/2)**2, disc=100):
        h=1.0/disc
        self.wpp_w=(w(h)-2*w(0)+w(-h))/h**2
        self.LOTW=LOTW
        self.w=w
        self.M=M
    def build(self, Lambdas=np.linspace(-2.0, 4.0, 100), Reds=10**np.linspace(2.0, 6.0, 100), disc=100, \
        strategy=lambda x: x, Ut_initguess=0.1):
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
                    M=self.M, LOTW=self.LOTW, outter_profile=self.w, disc=disc, strategy=strategy)
        
        rr, ll=np.meshgrid(Reds, Lambdas)
        fig=plt.figure()
        ax=plt.axes(projection='3d')
        ax.plot_surface(np.log10(rr), thxx, Cf)
        plt.xlabel('$log_{10}(Re_\delta)$')
        plt.ylabel('$\Lambda$')
        plt.show()

clsr=closure()
clsr.build()