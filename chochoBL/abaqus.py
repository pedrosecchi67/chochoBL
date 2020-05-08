import numpy as np
import scipy.interpolate as sinterp
import scipy.optimize as sopt
import closure as clsr

class turbulence_abaqus:
    def __init__(self, deltastars, props):
        self.rule=sinterp.UnivariateSpline(deltastars, props, ext=0) #engage extrapolation
        self.deltastar_lims=(deltastars[0], deltastars[1])
    def __call__(self, deltastar, dx=0):
        return self.rule(deltastar, nu=dx)

def Ksi(foo, disc=100):
    return np.trapz(foo(np.linspace(0.0, 1.0, disc)))

class Ksi_abaqus(turbulence_abaqus):
    def __init__(self, deltastars, foo=lambda x: x, disc=100): #foo necessarily takes arguments eta and deltastar
        props=np.zeros(len(deltastars))
        for i in range(len(deltastars)):
            props[i]=Ksi(foo=lambda eta: foo(eta, deltastars[i]), disc=disc)
        super().__init__(deltastars, props)
    def __call__(self, deltastar, dx=0):
        return super().__call__(deltastar, dx=dx)

class closure_abaqus:
    def __init__(self, LOTW=clsr.Gersten_Herwig_LOTW, w=lambda x: (1.0-np.cos(np.pi*x))/2, disc=100):
        self.disc=disc
        self.LOTW=LOTW
        self.w=w
        h=1.0/disc
        self.d2w_deta2=(w(h)-2*w(0)+w(-h))/h**2
    def build(self, Lambdas, Reds, Ut_initguess=0.1):
        mat=np.zeros((len(Lambdas), len(Reds)))
        for i in range(len(Lambdas)):
            for j in range(len(Reds)):
                mat[i, j]=sopt.fsolve(lambda Ut: self.turb_iter(Ut, Lambdas[i], Reds[j]), Ut_initguess)[0]
        Cf_spline=sinterp.RectBivariateSpline(Lambdas, Reds, mat**2*2)
        Ut_spline=sinterp.RectBivariateSpline(Lambdas, Reds, mat)
    def turb_iter(self, Ut, Lambda, Red):
        deltastar=Ut*Red
        A=Lambda/(Ut*self.d2w_deta2)
        return Ut-1.0/(self.LOTW(deltastar)+A*self.w(1.0))