import numpy as np
import numpy.linalg as lg
import scipy.optimize as sopt
import scipy.sparse as sps
import scipy.sparse.linalg as splg

import mapping
from residual import residual

'''
Module containing ADTs necessary for an iterative solution using scipy implementations
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

        self.mappings={
            'n':mapping.identity_mapping,
            'th11':mapping.identity_mapping,
            'H':mapping.identity_mapping, #mapping.sigma_mapping((1.0, 7.5)),
            'beta':mapping.identity_mapping, # mapping.sigma_mapping((-np.pi/2, np.pi/2)),
            'N':mapping.identity_mapping
        }

        self._inord=_base_inord.copy()

        self.inds={
            k:(i*self.msh.nnodes, (i+1)*self.msh.nnodes) for i, k in enumerate(self._inord)
        }

        self.scaler=scaler
        self.nit=0

        self.echo=echo

    def _fromx_extract(self, x, prop):
        indtup=self.inds[prop]

        return x[indtup[0]:indtup[1]]

    def arrmap(self, x):
        '''
        Transform an input array to auxiliary coordinates using mappings
        '''

        arrlist=[]

        for p in self._inord:
            inds=self.inds[p]

            arrlist.append(self.mappings[p].inv(x[inds[0]:inds[1]]))

        return np.hstack(arrlist)

    def arrunmap(self, x):
        '''
        Transform an array from auxiliary coordinates to proper greatnesses using mappings
        '''

        arrlist=[]
        gradlist=[]

        for p in self._inord:
            inds=self.inds[p]

            v, g=self.mappings[p](x[inds[0]:inds[1]])

            arrlist.append(v)
            gradlist.append(g)

        return np.hstack(arrlist), np.hstack(gradlist)

    def fun(self, x, qx, qy, qz, translate=False, calcjac=False):
        '''
        Calculate residuals
        '''

        self.x=(x.copy() if not translate else self.arrunmap(x)[0])
        self.qx=qx.copy()
        self.qy=qy.copy()
        self.qz=qz.copy()

        args=(
            self._fromx_extract(x, 'n'),
            self._fromx_extract(x, 'th11'),
            self._fromx_extract(x, 'H'),
            self._fromx_extract(x, 'beta'),
            self._fromx_extract(x, 'N'),
            qx, qy, qz
        )

        retvals=self.msh.mesh_getresiduals(*args)

        if calcjac:
            self.rmassj, self.rmomxj, self.rmomzj, self.renj, self.rtsj=self.msh.mesh_getresiduals_jac(*args)

        return np.hstack(retvals)

    def _get_dseeds(self, xd, translate=False):
        '''
        Execute direct AD subroutine to obtain dot product with Jacobian:
        for Krylov solver only
        '''

        x=self.x

        if translate:
            umx, gm=self.arrunmap(xd)
            mx=gm*umx
        else:
            mx=xd

        n, th11, h, beta, nts= \
            self._fromx_extract(x, 'n'), self._fromx_extract(x, 'th11'), self._fromx_extract(x, 'H'), self._fromx_extract(x, 'beta'), self._fromx_extract(x, 'N')

        nd, th11d, hd, betad, ntsd= \
            self._fromx_extract(mx, 'n'), self._fromx_extract(mx, 'th11'), self._fromx_extract(mx, 'H'), self._fromx_extract(mx, 'beta'), self._fromx_extract(mx, 'N')

        return np.hstack(self.msh.mesh_getresiduals_d(n, th11, h, beta, nts, self.qx, self.qy, self.qz, \
            nd=nd, th11d=th11d, hd=hd, betad=betad, ntsd=ntsd))

    def _Jaceval(self, J, vs):
        return sum([residual.jac_mult(J[:, :, :, i], self.msh.cellmatrix, v) for i, v in enumerate(vs)])

    def _get_dseeds_jac(self, xd, translate=False):
        '''
        Get derivative seeds from Jacobian
        '''

        if translate:
            umx, gm=self.arrunmap(xd)
            mx=gm*umx
        else:
            mx=xd

        perts= \
            (self._fromx_extract(mx, 'n'), self._fromx_extract(mx, 'th11'), self._fromx_extract(mx, 'H'), self._fromx_extract(mx, 'beta'), self._fromx_extract(mx, 'N'))

        return np.hstack([self._Jaceval(self.rmassj, perts), self._Jaceval(self.rmomxj, perts), self._Jaceval(self.rmomzj, perts), \
            self._Jaceval(self.renj, perts), self._Jaceval(self.rtsj, perts)])

    def Jac_convert(self, jacs):
        '''
        Convert Jacobians from a tuple into a full, sparse COO matrix
        '''

        # creating fundamental coordinates

        rows_base=np.hstack([
            cell.repeat(4) for cell in self.msh.cellmatrix-1
        ])
        cols_base=np.hstack([
            np.tile(cell, 4) for cell in self.msh.cellmatrix-1
        ])

        rows=[]
        cols=[]
        data=[]

        for i, resj in enumerate(jacs):
            for j in range(len(self._inord)):
                rows.append(rows_base+i*self.msh.nnodes)
                cols.append(cols_base+j*self.msh.nnodes)
                data.append(resj[:, :, :, j].flatten())

        nvars=len(self._inord)*self.msh.nnodes

        return sps.coo_matrix((np.hstack(data), (np.hstack(rows), np.hstack(cols))), (nvars, nvars))

    def get_jac_Krylov(self, x, qx, qy, qz, translate=False, usejac=True, fulljac=True):
        '''
        Generate linear operator to serve as Krylov space linear solver
        '''

        self.residuals=self.fun(x, qx, qy, qz, translate=translate, calcjac=True)

        if fulljac:
            oper=self.Jac_convert((self.rmassj, self.rmomxj, self.rmomzj, self.renj, self.rtsj))

        else:
            def _matvec(xd):
                if usejac:
                    retval=self._get_dseeds_jac(xd, translate=translate)
                else:
                    retval=self._get_dseeds(xd, translate=translate)
                retval[self.msh.CC+4*self.msh.nnodes]=xd[self.msh.CC+4*self.msh.nnodes]

                return retval

            oper=splg.LinearOperator((self.msh.nnodes*5, self.msh.nnodes*5), matvec=_matvec)

        return oper

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
