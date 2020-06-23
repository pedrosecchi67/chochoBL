import numpy as np
import numpy.linalg as lg
import scipy.optimize as sopt

'''
Module containing ADTs necessary for a conjugate gradient descent using scipy
'''

import adjoint as adj

_base_inord=['n', 'th11', 'H', 'beta', 'N'] # EIF modelling not yet included

def total_residual(value):
    return sum(v@v for v in value.values())/2

class optunit:
    '''
    Class containing x, f(x) and gradf(x), stored for optimization module
    '''

    def __init__(self, msh, scaler=1.0, echo=False):
        self.msh=msh

        self._inord=_base_inord.copy()

        if not self.msh.CC is None:
            self._inord.remove('N')

            self.has_transadjoint=True
        else:
            self.has_transadjoint=False

        self.inds={
            k:(i*self.msh.nnodes, (i+1)*self.msh.nnodes) for i, k in enumerate(self._inord)
        }

        self.scaler=scaler
        self.nit=0

        self.echo=echo

    def _fromx_extract(self, prop):
        indtup=self.inds[prop]

        return self.x[indtup[0]:indtup[1]]

    def set_value(self, x, N=None):
        '''
        Set the value of x as a given vector
        '''

        self.x=x.copy()

        indict={inp:self._fromx_extract(inp) for inp in self._inord}

        indict.update(self.q)

        if not N is None:
            indict.update({'N':N})

        # set in mesh
        self.msh.set_values(
            indict, nodal=False
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

        self.set_value(x, N=np.zeros(self.msh.nnodes) if self.has_transadjoint else None)

        if self.has_transadjoint:
            # solve adjoint system for transition
            value, grad=self.msh.calculate_graph(ends=['RTS'])

            distJ=self.msh.dcell_dnode[1]

            derivs_dir=self.msh.gr.get_derivs_direct('N', ends=['RTS'])

            A=distJ.T@derivs_dir['RTS']
            b=distJ.T@value['RTS']

            N, _=adj.solve_0CC(A, -b, self.msh.CC)

            self.msh.set_values({'N':N}, nodes=['N'], nodal=False, reset=False)

            value, grad=self.msh.calculate_graph()

            b=grad['N']

            psi, self.msh.trans_prec_SuperLU=adj.solve_0CC(A.T, -b, self.msh.CC)
            psi=distJ@psi

            derivs_rev=self.msh.gr.get_derivs_reverse('RTS')

            grad={p:grad[p]+(psi@derivs_rev[p] if not derivs_rev[p] is None else 0.0) for p in self._inord}
        else:
            value, grad=self.msh.calculate_graph()

        self.fx=total_residual(value)*self.scaler
        self.grad=np.hstack([grad[p] for p in self._inord])*self.scaler

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

        return np.hstack([x[p] for p in self._inord])

    def solve(self, x0, q={}, solinfo=False, relgtol=1e-2, maxiter=200, method='CG', w=1.0, b=0.5, \
        init_alpha=1e-3, factor_echo=False):
        '''
        Solve boundary layer equations via iterative methods, Conjugate Gradients as default
        '''

        self.q=q

        initguess_vector=self.pack(x0)

        if method=='custom_adaptative':
            xans, fans, _, nit, success=custom_adaptative(itermax=maxiter, relgtol=relgtol, fun=self.fun, jac=self.jac, \
                x0=initguess_vector, w=w, b=b, init_alpha=init_alpha, \
                    factor_echo=factor_echo)
        else:
            ng0=lg.norm(self.jac(initguess_vector))

            soln=sopt.minimize(fun=self.fun, x0=initguess_vector, jac=self.jac, method=method, \
                options={'maxiter':maxiter, 'gtol':relgtol*ng0})

            if not soln.success:
                print(soln)
                raise Exception('Iterative solution failed')

            xans=soln.x

        solution={}

        for p in self._inord:
            inds=self.inds[p]

            solution[p]=xans[inds[0]:inds[1]]

        if self.has_transadjoint:
            solution['N']=self.msh.gr.get_value('N')[0]

        if solinfo:
            if method=='custom_adaptative':
                return solution, nit, success
            else:
                return solution, soln.nit, soln.success
        return solution

def custom_adaptative(fun, jac, x0, init_alpha=1.0, w=1.0, b=0.5, itermax=100, relgtol=1e-3, factor_echo=False):
    '''
    Customized gradient descent algorithm
    '''

    x=x0.copy()

    f0=fun(x)
    g0=jac(x)

    glast=g0.copy()
    g=g0.copy()
    f=f0

    ng02=g0@g0

    nit=0

    alpha=init_alpha/np.sqrt(ng02)

    unconverged=True

    while nit<=itermax and unconverged:
        x-=alpha*g

        f=fun(x)
        g=jac(x)

        nglast2=(glast@glast)

        factor=(g@glast)/(1e-20 if nglast2<1e-20 else nglast2)

        alpha*=np.exp(w*(factor-b))

        if factor_echo:
            print('%5s %5s %5s %5s'% ('fac', 'alph', 'f', 'ng'))
            print('%5g %5g %5g %5g'% (factor, alpha, f, np.sqrt(nglast2)))

        nit+=1

        glast=g.copy()

        unconverged=unconverged and nglast2>ng02*relgtol**2

    if nit<=itermax:
        success=not unconverged
    else:
        success=False

    return x, f, g, nit, success
