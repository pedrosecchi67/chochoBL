import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.interpolate as sinterp
import time as tm
import abaqus as abq

Gersten_Herwig_A=6.1e-4
Gersten_Herwig_B=1.43e-3
Gersten_Herwig_Lambda=(Gersten_Herwig_A+Gersten_Herwig_B)**(1.0/3)
Von_Karman_kappa=0.41
def Gersten_Herwig_LOTW(eta, deltastar):
    return (np.log((Gersten_Herwig_Lambda*deltastar*eta+1.0)/np.sqrt((Gersten_Herwig_Lambda*eta*deltastar)**2-\
        Gersten_Herwig_Lambda*eta*deltastar+1.0))/3+(np.atan((2*Gersten_Herwig_Lambda*eta*deltastar-1.0)/np.sqrt(3))+np.pi/6)/np.sqrt(3))/Gersten_Herwig_Lambda+\
            np.log(1+Von_Karman_kappa*Gersten_Herwig_B*(deltastar*eta)**4)/(4*Von_Karman_kappa)

class closure:
    def __init__(self, M=lambda eta: 1.0-eta**2, a=lambda eta: 2*eta-2*eta**3+eta**4, \
        b=lambda eta: eta*(1.0-eta**3), LOTW=Gersten_Herwig_LOTW):
        pass