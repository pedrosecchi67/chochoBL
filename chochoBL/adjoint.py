import numpy as np
import scipy.optimize as sopt

'''
Module containing functions pertinent to adjoint method
'''

def _R_dRdp(A, p, b):
    r=A@p-b

    return r@r/2, r@A

class optimcore:
    '''
    Class containing fun and jac to guarantee function executions won't be repeated
    '''

    def __init__(self, fg):
        self._fg=fg

    def setvalue(self, x):
        self.x=x

    def compvalue(self, x):
        if not hasattr(self, 'x'):
            return False

        return all(self.x==x)

    def calculate(self):
        self.fx, self.grad=self._fg(self.x)

    def fun(self, x):
        if not self.compvalue(x):
            self.setvalue(x)
            self.calculate()

        return self.fx

    def jac(self, x):
        if not self.compvalue(x):
            self.setvalue(x)
            self.calculate()

        return self.grad

def sys_solve(A, b, x0=None):
    '''
    Function to solve a linear system using conjugate gradient method applied to least squares residual
    '''

    opt=optimcore(fg=lambda p: _R_dRdp(A, p, b))

    return sopt.minimize(fun=opt.fun, jac=opt.jac, x0=(np.zeros_like(b) if x0 is None else x0), method='CG', options={'gtol':1e-3})
