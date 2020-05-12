import numpy as np
import scipy.interpolate as sinterp
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class abaqus:
    def __init__(self, x, y, z):
        self.rule=sinterp.interp2d(x, y, z, bounds_error=False, kind='cubic')

        self.xlims=[np.amin(x), np.amax(x)]
        self.ylims=[np.amin(y), np.amax(y)]

    def __call__(self, x, y, dx=0, dy=0):
        
        return self.rule(x, y, dx=dx, dy=dy)[0]

class abaqus_1d:
    def __init__(self, x, y):
        self.rule=sinterp.UnivariateSpline(x, y)

        self.xlims=[np.amin(x), np.amax(x)]
    
    def __call__(self, x, dx=0):
        
        return self.rule(x, nu=dx)
