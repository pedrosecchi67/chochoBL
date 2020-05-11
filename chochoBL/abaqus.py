import numpy as np
import scipy.interpolate as sinterp
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class abaqus:
    def __init__(self, x, y, z, fun, funp, initguess):
        soln=sopt.minimize(lambda pars: np.sum((fun(x, y, pars)-z)**2), x0=initguess, method='Nelder-Mead', options={'maxiter':1e5}, tol=1e-5)
        self.success=soln.success
        self.pars=soln.x
        self.fun=fun
        self.funp=funp
    def __call__(self, x, y, dx=0, dy=0):
        return self.fun(x, y, self.pars)
