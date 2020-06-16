import numpy as np
import numpy.linalg as lg
import scipy.optimize as sopt

'''
Module containing ADTs necessary for a conjugate gradient descent using scipy
'''

_inord=['n', 'th11', 'H', 'beta', 'N'] # EIF modelling not yet included

def total_residual(value):
    return sum(v@v for v in value.values())/2

class optunit:
    '''
    Class containing x, f(x) and gradf(x), stored for optimization module
    '''

    def __init__(self, msh, scaler=1.0, echo=False):
        self.msh=msh

        self.inds={
            'n':(0, msh.nnodes),
            'th11':(msh.nnodes, 2*msh.nnodes),
            'H':(2*msh.nnodes, 3*msh.nnodes),
            'beta':(3*msh.nnodes, 4*msh.nnodes),
            'N':(4*msh.nnodes, 5*msh.nnodes)
        }

        self.scaler=scaler
        self.nit=0

        self.echo=echo

    def _fromx_extract(self, prop):
        indtup=self.inds[prop]

        return self.x[indtup[0]:indtup[1]]

    def set_value(self, x):
        '''
        Set the value of x as a given vector
        '''

        self.x=x

        # set in mesh
        self.msh.set_values(
            {
                'n':{'n':self._fromx_extract('n')},
                'th11':{'th11':self._fromx_extract('th11')},
                'H':{'H':self._fromx_extract('H')},
                'beta':{'beta':self._fromx_extract('beta')},
                'N':{'N':self._fromx_extract('N')}, 
                'q':self.q
            }
        )

    def x_compare(self, x):
        '''
        Check whether the requested value of x has already been set
        '''

        if not hasattr(self, 'x'):
            self.x=x

            return False

        return np.all(x==self.x)
    
    def calculate(self, x):
        '''
        Run graph calculation
        '''

        self.set_value(x)

        value, grad=self.msh.calculate_graph()

        self.fx=total_residual(value)*self.scaler
        self.grad=np.hstack([grad[p] for p in _inord])*self.scaler

        self.nit+=1

        if self.echo:
            print('Iteration: ', self.nit, ' norm of gradient: ', lg.norm(self.grad), ' fun: ', self.fx)

    def fun(self, x):
        '''
        Objective function (total residual)
        '''

        if not self.x_compare(x):
            self.calculate(x)

        return self.fx

    def jac(self, x):
        '''
        Jacobian function (for total residual, returned as 1-D array)
        '''

        if not self.x_compare(x):
            self.calculate(x)

        return self.grad

    def pack(self, x):
        '''
        Pack a dictionary into a single input array
        '''

        return np.hstack([x[p] for p in _inord])

    def solve(self, x0, q={}, solobj=False, relgtol=1e-2, maxiter=200, method='CG'):
        '''
        Solve boundary layer equations via iterative methods, Conjugate Gradients as default
        '''

        self.q=q

        initguess_vector=self.pack(x0)

        ng0=lg.norm(self.jac(initguess_vector))

        soln=sopt.minimize(fun=self.fun, x0=initguess_vector, jac=self.jac, method=method, \
            options={'maxiter':maxiter, 'gtol':relgtol*ng0})

        if not soln.success:
            print(soln)
            raise Exception('Iterative solution failed')

        solution={}

        for p in _inord:
            inds=self.inds[p]

            solution[p]=soln.x[inds[0]:inds[1]]

        if solobj:
            return solution, soln
        else:
            return solution
