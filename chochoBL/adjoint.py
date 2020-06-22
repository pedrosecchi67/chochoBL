import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as slg
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

def sys_solve(A, b, x0=None, method='CG', inv=None, gtol=1e-3):
    '''
    Function to solve a linear system using conjugate gradient method applied to least squares residual
    '''

    if method=='analytic':
        if inv is None:
            inv=slg.splu(A)
            return inv.solve(b), inv
        return inv.solve(b)

    if method=='CG_iter':
        soln=slg.cg(A, b, x0=x0 if inv is None else inv.solve(b))

        if soln[1]:
            print(soln)
            raise Exception('Iterative solution of linear system in adjoint problem failed.')

        return soln[0]

    if method=='lsqr_iter':
        soln=slg.lsqr(A, b, x0=x0 if inv is None else inv.solve(b), atol=gtol, btol=gtol)

        if not soln[1]:
            print(soln)
            raise Exception('Iterative solution of linear system in adjoint problem failed.')

        return soln[0]

    opt=optimcore(fg=lambda p: _R_dRdp(A, p, b))

    initguess=((np.zeros_like(b) if x0 is None else x0) if inv is None else inv.solve(b))

    soln=sopt.minimize(fun=opt.fun, jac=opt.jac, x0=initguess, \
        method='CG', options={'gtol':gtol})

    if not soln.success:
        print(soln)
        raise Exception('Iterative solution of linear system in adjoint problem failed.')

    return soln.x

def solve_0CC(A, b, CCs, method='analytic', x0=None, inv=None):
    '''
    Solve a linear system setting contour conditions of x[CCs]=0.0, for system Ax=b
    '''

    Ap=A.copy()

    if not isinstance(Ap, sps.csr_matrix):
        Ap=Ap.tocsr()

    nCCs=np.setdiff1d(np.arange(np.size(b, axis=0), dtype='int'), CCs)

    Ap=Ap[nCCs, :]
    Ap=Ap.tocsc()[:, nCCs]

    bp=b[nCCs]

    if inv is None and method=='analytic':
        xp, inv=sys_solve(Ap, bp, method=method, x0=x0, inv=inv)

        x=np.zeros_like(b)
        x[nCCs]=xp

        return x, inv
    else:
        xp=sys_solve(Ap, bp, method=method, x0=x0, inv=inv)

        x=np.zeros_like(b)
        x[nCCs]=xp

        return x
