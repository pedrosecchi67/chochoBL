import numpy as np
import scipy.sparse as sps

def add_set(l, s):
    '''
    Add set s to list l if not yet present
    '''

    if not s in l:
        l.append(s)
        return len(l)-1
    else:
        return l.index(s)

#exponents for polynomial basis for auxiliary coordinates in 2D finite element
_Bp_exponents=np.array(
    [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3], 
        [1, 0], 
        [1, 1],
        [1, 2], 
        [1, 3],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3]
    ]
)

def _get_Bp_index(nksi, neta):
    '''
    Return index of given ksi, eta combination in Bp basis
    '''

    return 4*nksi+neta

def _get_Bp_vector(nksi, neta):
    '''
    Return vector in base Bp given exponents for auxiliary coordinates in 2D domain
    '''

    v=np.zeros(16)
    v[_get_Bp_index(nksi, neta)]=1.0

    return v

def _get_Bp_integral_matrix():
    '''
    Deduce integral matrix S, Sp=int(int(p, ksi), eta) in [0, 1]x[0, 1] domain
    '''

    S=np.zeros(16)

    for i in range(16):
        S[i]=(1.0/(_Bp_exponents[i, 0]+1))*(1.0/(_Bp_exponents[i, 1]+1))
    
    return S

_Bp_integral_matrix=_get_Bp_integral_matrix()

def _get_Bp_product_matrix():
    '''
    Deduce three-dimensional matrix to compute the product between two Bp basis 
    polynomials as a bilinear transformation
    '''

    P=np.zeros((16, 16, 16))

    for i, p1_exps in enumerate(list(_Bp_exponents)):
        for j, p2_exps in enumerate(list(_Bp_exponents)):
            total_exps=[p1_exps[0]+p2_exps[0], p1_exps[1]+p2_exps[1]]

            total_index=_get_Bp_index(total_exps[0], total_exps[1])

            if total_index<16:
                P[total_index, i, j]+=1.0
    
    return P

_Bp_product_matrix=_get_Bp_product_matrix()

def _get_Mbp_matrix():
    '''
    Generates matrix for conversion between Bp and Bn bases
    '''

    Mbp=np.zeros((16, 4))

    #polynomial 1-ksi, 1-eta
    ksi1=np.zeros(16)
    eta1=np.zeros(16)
    
    ksi1[_get_Bp_index(0, 0)]=1.0
    ksi1[_get_Bp_index(1, 0)]=-1.0

    eta1[_get_Bp_index(0, 0)]=1.0
    eta1[_get_Bp_index(0, 1)]=-1.0

    #polynomial ksi, eta
    ksi2=np.zeros(16)
    eta2=np.zeros(16)
    
    ksi2[_get_Bp_index(1, 0)]=1.0

    eta2[_get_Bp_index(0, 1)]=1.0

    #N1=(1.0-ksi)*(1.0-eta)
    N1=ksi1@_Bp_product_matrix@eta1

    #N2=ksi*(1.0-eta)
    N2=ksi2@_Bp_product_matrix@eta1

    #N3=ksi*eta
    N3=ksi2@_Bp_product_matrix@eta2

    #N1=(1.0-ksi)*eta
    N4=ksi1@_Bp_product_matrix@eta2

    Mbp[:, 0]=N1
    Mbp[:, 1]=N2
    Mbp[:, 2]=N3
    Mbp[:, 3]=N4

    return Mbp

_Mbp=_get_Mbp_matrix()

def _get_Bp_derivative_matrix():
    '''
    Returns derivative matrixes [d/dksi]{p}={dp/dksi}, [d/deta]{p}={dp/deta}
    '''

    dksi=np.zeros((16, 16))
    deta=np.zeros((16, 16))

    for i, exps in enumerate(list(_Bp_exponents)):
        dksi_exps=[exps[0]-1, exps[1]]
        deta_exps=[exps[0], exps[1]-1]

        #disconsider derivatives of constants
        for j, d in enumerate(dksi_exps):
            dksi_exps[j]=max([0, d])
        for j, d in enumerate(deta_exps):
            deta_exps[j]=max([0, d])
        
        dksi[_get_Bp_index(dksi_exps[0], dksi_exps[1]), i]+=exps[0]
        deta[_get_Bp_index(deta_exps[0], deta_exps[1]), i]+=exps[1]
    
    return dksi, deta

_Bp_dksi, _Bp_deta=_get_Bp_derivative_matrix()

def get_detJacobian(xs, ys):
    '''
    Return determinant of Jacobian as a Bp polynomial vector according to
    {x} and {y} inputs.
    '''

    px=_Mbp@xs
    py=_Mbp@ys

    detJ=(_Bp_dksi@px)@_Bp_product_matrix@(_Bp_deta@py)-(_Bp_dksi@py)@_Bp_product_matrix@(_Bp_deta@px)

    return detJ

def surface_integral_matrix(xs, ys):
    '''
    Returns matrix A such that, v being a linearized property in base Bn, 
    int(int(v, x), y)=[A]{v}
    '''

    detJ=get_detJacobian(xs, ys)

    Smat=_Bp_integral_matrix@np.swapaxes(_Bp_product_matrix, 0, 1)

    return detJ@Smat

_Bp_Smat=_Bp_integral_matrix@np.swapaxes(_Bp_product_matrix, 0, 1)

def udvdx_residual_matrix(xs, ys):
    '''
    Returns matrix A (three-dimensional) such that, being u and v properties,
    {r}={u}[A]{v}, r_i=int(int(N_i*u*dv_dx))
    This expression is made so as to ease the generation of divergent residuals
    '''

    py=_Mbp@ys

    dy_dksi=_Bp_dksi@py
    dy_deta=_Bp_deta@py
    
    resmat=np.zeros((4, 4, 4))

    A=_Mbp.T@_Bp_Smat
    B=_Bp_product_matrix@(dy_deta@_Bp_product_matrix@_Bp_dksi-dy_dksi@_Bp_product_matrix@_Bp_deta)@_Mbp
    resmat=A@np.swapaxes(_Mbp.T@B, 0, 1)

    return resmat

def udvdy_residual_matrix(xs, ys):
    '''
    Returns matrix A (three-dimensional) such that, being u and v properties,
    {r}={u}[A]{v}, r_i=int(int(N_i*u*dv_dy))
    This expression is made so as to ease the generation of divergent residuals
    '''

    px=_Mbp@xs

    dx_dksi=_Bp_dksi@px
    dx_deta=_Bp_deta@px
    
    resmat=np.zeros((4, 4, 4))

    A=_Mbp.T@_Bp_Smat
    B=_Bp_product_matrix@(-dx_deta@_Bp_product_matrix@_Bp_dksi+dx_dksi@_Bp_product_matrix@_Bp_deta)@_Mbp
    resmat=A@np.swapaxes(_Mbp.T@B, 0, 1)

    return resmat

def dvdx_residual_matrix(xs, ys):
    '''
    Returns matrix A such that, being v a property,
    {r}=[A]{v}, r_i=int(int(N_i*dv_dx))
    This expression is made so as to ease the generation of divergent residuals
    '''

    py=_Mbp@ys

    dy_dksi=_Bp_dksi@py
    dy_deta=_Bp_deta@py

    return (_Mbp.T@_Bp_product_matrix@(dy_deta@_Bp_product_matrix@_Bp_dksi-dy_dksi@_Bp_product_matrix@_Bp_deta)@_Mbp).T@_Bp_integral_matrix

def dvdy_residual_matrix(xs, ys):
    '''
    Returns matrix A such that, being v a property,
    {r}=[A]{v}, r_i=int(int(N_i*dv_dx))
    This expression is made so as to ease the generation of divergent residuals
    '''

    px=_Mbp@xs

    dx_dksi=_Bp_dksi@px
    dx_deta=_Bp_deta@px

    return (_Mbp.T@_Bp_product_matrix@(-dx_deta@_Bp_product_matrix@_Bp_dksi+dx_dksi@_Bp_product_matrix@_Bp_deta)@_Mbp).T@_Bp_integral_matrix

def v_residual_matrix(xs, ys):
    '''
    Returns matrix A such that, being v a property,
    {r}=[A]{v}, r_i=int(int(v*N_i))
    This expression is made so as to ease the generation of residuals
    '''

    Smat=surface_integral_matrix(xs, ys)
    A=(_Mbp.T@_Bp_product_matrix@_Mbp)
    
    return Smat@np.swapaxes(A, 0, 1)