import numpy as np
import scipy.interpolate as sinterp
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class AbaqusBoundsError(Exception):
    pass

class abaqus:
    def __init__(self, x, y, z):
        self.rule=sinterp.RectBivariateSpline(x, y, z)

        self.xlims=[np.amin(x), np.amax(x)]
        self.ylims=[np.amin(y), np.amax(y)]

    def __call__(self, x, y, dx=0, dy=0):
        if x<self.xlims[0] or x>self.xlims[1]:
            raise AbaqusBoundsError('x value out of abaqus bounds')
        elif y<self.ylims[0] or y>self.ylims[1]:
            raise AbaqusBoundsError('y value out of abaqus bounds')
        
        return self.rule(x, y, dx=dx, dy=dy)
