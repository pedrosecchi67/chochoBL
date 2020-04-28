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
        
        self.a=a; self.b=b; self.M=M; self.LOTW=LOTW

        #Defining turbulence dependant Ksis
        self.Ksi_W=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar), disc=Ksi_disc)
        self.Ksi_W2=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)**2, disc=Ksi_disc)
        self.Ksi_Wa=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*a(eta), disc=Ksi_disc)
        self.Ksi_Wb=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*b(eta), disc=Ksi_disc)
        self.Ksi_WM=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*M(eta), disc=Ksi_disc)
        self.Ksi_WMa=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*M(eta)*a(eta), disc=Ksi_disc)
        self.Ksi_WMb=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)*M(eta)*b(eta), disc=Ksi_disc)
        self.Ksi_W2M=abq.Ksi_abaqus(deltastars, foo=lambda eta, deltastar: LOTW(eta*deltastar)**2*M(eta), disc=Ksi_disc)
        
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

        self.atmosphere=atmosphere

        h=1.0/Ksi_disc
        self.ap_w=(a(h)-a(0.0))/h
        self.bp_w=(b(h)-b(0.0))/h