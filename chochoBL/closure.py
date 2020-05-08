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

class closure:
    def __init__(self, M=lambda eta: 1.0-eta**2, a=lambda eta: 2*eta-2*eta**3+eta**4, \
        b=lambda eta: eta*(1.0-eta**3)/6, LOTW=Gersten_Herwig_LOTW, deltastar_lims=[0.0, 500.0], \
            deltastar_rule=lambda x: (np.sin((x-0.5)*np.pi)+1.0)/2, deltastar_disc=50, Ksi_disc=100, atmosphere=atm.ATMOSPHERE_1976(Z=0.0, dT=0.0)):
        deltastars=np.interp(deltastar_rule(np.linspace(0.0, 1.0, deltastar_disc)), [0.0, 1.0], deltastar_lims)
        
        self.a=a; self.b=b; self.M=M; self.LOTW=LOTW; self.Ksi_disc=Ksi_disc

        #Defining turbulence dependant Ksis
        self.Ksi_W=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar), disc=Ksi_disc)
        self.Ksi_W2=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)**2, disc=Ksi_disc)
        self.Ksi_Wa=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*a(eta), disc=Ksi_disc)
        self.Ksi_Wb=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*b(eta), disc=Ksi_disc)
        self.Ksi_WM=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*M(eta), disc=Ksi_disc)
        self.Ksi_WMa=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*M(eta)*a(eta), disc=Ksi_disc)
        self.Ksi_WMb=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*M(eta)*b(eta), disc=Ksi_disc)
        self.Ksi_W2M=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)**2*M(eta), disc=Ksi_disc)
        self.Ksi_W2M2=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)**2*M(eta)**2, disc=Ksi_disc)
        self.Ksi_WM2a=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*M(eta)**2*a(eta), disc=Ksi_disc)
        self.Ksi_WM2b=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*M(eta)**2*b(eta), disc=Ksi_disc)
        
        #defining constant ksis
        self.Ksi_a=abq.Ksi(foo=a, disc=Ksi_disc)
        self.Ksi_b=abq.Ksi(foo=b, disc=Ksi_disc)
        self.Ksi_ab=abq.Ksi(foo=lambda eta: a(eta)*b(eta), disc=Ksi_disc)
        self.Ksi_a2=abq.Ksi(foo=lambda eta: a(eta)**2, disc=Ksi_disc)
        self.Ksi_b2=abq.Ksi(foo=lambda eta: b(eta)**2, disc=Ksi_disc)
        self.Ksi_Ma=abq.Ksi(foo=lambda eta: M(eta)*a(eta), disc=Ksi_disc)
        self.Ksi_Mb=abq.Ksi(foo=lambda eta: M(eta)*b(eta), disc=Ksi_disc)
        self.Ksi_Mab=abq.Ksi(foo=lambda eta: M(eta)*a(eta)*b(eta), disc=Ksi_disc)
        self.Ksi_Ma2=abq.Ksi(foo=lambda eta: M(eta)*a(eta)**2, disc=Ksi_disc)
        self.Ksi_Mb2=abq.Ksi(foo=lambda eta: M(eta)*b(eta)**2, disc=Ksi_disc)
        self.Ksi_M2a2=abq.Ksi(foo=lambda eta: M(eta)**2*a(eta)**2, disc=Ksi_disc)
        self.Ksi_M2b2=abq.Ksi(foo=lambda eta: M(eta)**2*b(eta)**2, disc=Ksi_disc)
        self.Ksi_M2ab=abq.Ksi(foo=lambda eta: M(eta)**2*a(eta)*b(eta), disc=Ksi_disc)

        self.atmosphere=atmosphere

        h=1.0/Ksi_disc
        self.ap_w=(a(h)-a(0.0))/h
        self.bp_w=(b(h)-b(0.0))/h

def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

def poly2Dreco(X, Y, c):
    '''
    Evaluates polyfit2d
    '''

    return (c[0] + X*c[1] + Y*c[2] + X**2*c[3] + X**2*Y*c[4] + X**2*Y**2*c[5] + 
           Y**2*c[6] + X*Y**2*c[7] + X*Y*c[8])

class closure_abaqus:
    def __init__(self, LOTW=Gersten_Herwig_LOTW, w=lambda x: (1.0-np.cos(np.pi*x))/2, disc=100):
        self.disc=disc
        self.LOTW=LOTW
        self.w=w
        h=1.0/disc
        self.d2w_deta2=(w(h)-2*w(0)+w(-h))/h**2
    def build(self, As=np.linspace(-2.0, 1000.0, 100), dsts=10.0**np.linspace(2.0, 6.0, 100), ad_plot=False, lr_plot=False, logz_plot=True):
        Lambda_mat=np.zeros((len(As), len(dsts)))
        Red_mat=np.zeros((len(As), len(dsts)))
        Cf_mat=np.zeros((len(As), len(dsts)))
        Ut_mat=np.zeros((len(As), len(dsts)))
        for i in range(len(As)):
            for j in range(len(dsts)):
                Lambda_mat[i, j], Red_mat[i, j], Cf_mat[i, j], Ut_mat[i, j]=self.calc(dst=dsts[j], A=As[i])
        if ad_plot:
            fig=plt.figure()
            ax=plt.axes(projection='3d')
            aa, dd=np.meshgrid(As, dsts)
            ax.plot_surface(aa, np.log10(dd), np.log10(Cf_mat) if logz_plot else Cf_mat)
            plt.show()
        if lr_plot:
            fig=plt.figure()
            ax=plt.axes(projection='3d')
            ax.plot_surface(np.exp(Lambda_mat), np.log10(Red_mat), np.log10(Cf_mat) if logz_plot else Cf_mat)
            plt.xlabel('$\Lambda$')
            plt.show()
    def calc(self, dst, A):
        ust_edge=self.LOTW(dst)+A*self.w(1.0)
        Red=ust_edge*dst
        Ut=dst/Red
        Lambda=self.d2w_deta2*A*Ut
        Cf=2.0/ust_edge**2
        return Lambda, Red, Cf, Ut

def_turb_clsr_abaq=closure_abaqus()
def_turb_clsr_abaq.build(lr_plot=True)