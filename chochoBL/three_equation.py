import numpy as np
import scipy.sparse as sps

'''
Module containing functions and classes referrant to the application of a three-equation
boundary layer solver.

Include in passive dictionary:
\'normal\' key with normal vectors, stacked (shape (npan, 3))
\'cells\' key with list of cells
'''

from differentiation import *

def qx_nodal_matrix(nnodes, cells):
    '''
    Function recieving a number of nodes and a set of cells and returning a set of
    (sparse) matrixes for the linear transformation:
    qx(nodal notation)=Tx.qx+Ty.qy+Tz.qz (inputted as raw vectors)
    '''

    ncells=len(cells)

    Tx=sps.lil_matrix((ncells*4, nnodes))
    Ty=sps.lil_matrix((ncells*4, nnodes))
    Tz=sps.lil_matrix((ncells*4, nnodes))

    for i, c in enumerate(cells):
        for j in range(4):
            Tx[4*i+j, c.indset[j]]=c.Mtosys[0, 0]
            Ty[4*i+j, c.indset[j]]=c.Mtosys[0, 1]
            Tz[4*i+j, c.indset[j]]=c.Mtosys[0, 2]
    
    return Tx, Ty, Tz

def qz_nodal_matrix(nnodes, cells):
    '''
    Function recieving a number of nodes and a set of cells and returning a set of
    (sparse) matrixes for the linear transformation:
    qz(nodal notation)=Tx.qx+Ty.qy+Tz.qz (inputted as raw vectors)
    '''

    ncells=len(cells)

    Tx=sps.lil_matrix((ncells*4, nnodes))
    Ty=sps.lil_matrix((ncells*4, nnodes))
    Tz=sps.lil_matrix((ncells*4, nnodes))

    for i, c in enumerate(cells):
        for j in range(4):
            Tx[4*i+j, c.indset[j]]=c.Mtosys[2, 0]
            Ty[4*i+j, c.indset[j]]=c.Mtosys[2, 1]
            Tz[4*i+j, c.indset[j]]=c.Mtosys[2, 2]
    
    return Tx, Ty, Tz

def uf(qx, qy, qz, passive):
    msh=passive['mesh']

    return msh.Juqx@qx+msh.Juqy@qy+msh.Juqz@qz

def duf_dqx(qx, qy, qz, passive):

    return passive['mesh'].Juqx

def duf_dqy(qx, qy, qz, passive):

    return passive['mesh'].Juqy

def duf_dqz(qx, qy, qz, passive):

    return passive['mesh'].Juqz

def wf(qx, qy, qz, passive):
    msh=passive['mesh']

    return msh.Jwqx@qx+msh.Jwqy@qy+msh.Jwqz@qz

def dwf_dqx(qx, qy, qz, passive):

    return passive['mesh'].Jwqx

def dwf_dqy(qx, qy, qz, passive):

    return passive['mesh'].Jwqy

def dwf_dqz(qx, qy, qz, passive):

    return passive['mesh'].Jwqz

def uw_conversion_getnode(msh):
    '''
    Add universal-to-local nodal velocity conversion matrixes and methods to a mesh and return a node for it\'s 
    qi-uw conversion
    '''

    nnodes=len(msh.nodes)
    ncells=len(msh.cells)

    msh.Juqx, msh.Juqy, msh.Juqz=qx_nodal_matrix(nnodes, msh.cells)
    msh.Jwqx, msh.Jwqy, msh.Jwqz=qz_nodal_matrix(nnodes, msh.cells)

    ufunc=func(f=uf, derivs=(duf_dqx, duf_dqy, duf_dqz,), args=[0, 1, 2], sparse=True, haspassive=True)
    wfunc=func(f=wf, derivs=(dwf_dqx, dwf_dqy, dwf_dqz,), args=[0, 1, 2], sparse=True, haspassive=True)
    
    uwfs=funcset(fs=[ufunc, wfunc], arglens=[nnodes, nnodes, nnodes], outlens=[4*ncells, 4*ncells], sparse=True)

    uwnode=node(f=uwfs, args_to_inds=['qx', 'qy', 'qz'], outs_to_inds=['u', 'w'], passive=msh.passive)

    return uwnode

def qef(qx, qy, qz):
    return np.sqrt(qx**2+qy**2+qz**2)

def dqef_dqx(qx, qy, qz):
    return sps.diags(qx/np.sqrt(qx**2+qy**2+qz**2), format='lil')

def dqef_dqy(qx, qy, qz):
    return sps.diags(qy/np.sqrt(qx**2+qy**2+qz**2), format='lil')

def dqef_dqz(qx, qy, qz):
    return sps.diags(qz/np.sqrt(qx**2+qy**2+qz**2), format='lil')

def qe_getnode(msh):
    '''
    Add qe (scalar) computation functions and return correspondent node, to be used with a mesh
    '''

    nnodes=len(msh.nodes)

    qefunc=func(f=qef, derivs=(dqef_dqx, dqef_dqy, dqef_dqz,), args=[0, 1, 2], sparse=True, haspassive=False)

    qenode=node(f=qefunc, args_to_inds=['qx', 'qy', 'qz'], outs_to_inds=['qe'], passive=msh.passive)

    return qenode

def Hf(th11, deltastar1):
    return deltastar1/th11

def dHf_dth11(th11, deltastar1):
    return sps.diags(-deltastar1/th11**2, format='lil')

def dHf_ddeltastar1(th11, deltastar1):
    return sps.diags(1.0/th11, format='lil')

def H_getnode(msh):
    '''
    Add H (streamwise shape parameter) computation functions and return correspondent node, 
    to be used with a mesh
    '''

    Hfunc=func(f=Hf, derivs=(dHf_dth11, dHf_ddeltastar1,), args=[0, 1], sparse=True, haspassive=False)

    Hnode=node(f=Hfunc, args_to_inds=['th11', 'deltastar1'], outs_to_inds=['H'], passive=msh.passive)

    return Hnode

def Mef(qe, passive):
    return qe/passive['atm'].v_sonic

def dMef_dqe(qe, passive):
    return sps.eye(len(qe), format='lil')/passive['atm'].v_sonic

def Me_getnode(msh):
    '''
    Add Me (external Mach number) computation functions and return correspondent node,
    to be used with a mesh
    '''

    Mefunc=func(f=Mef, derivs=(dMef_dqe,), args=[0], sparse=True, haspassive=True)

    Menode=node(f=Mefunc, args_to_inds=['qe'], outs_to_inds=['Me'], passive=msh.passive)

    return Menode

def rhof(Me, passive):
    a=passive['atm'].v_sonic
    rho0=passive['atm'].rho
    Uinf=passive['Uinf']

    return (1.0-Me**2*(Me*a-Uinf)/Uinf)*rho0

def drhof_dMe(Me, passive):
    a=passive['atm'].v_sonic
    rho0=passive['atm'].rho
    Uinf=passive['Uinf']

    return sps.diags((-Me**2*(a-Uinf)/Uinf-2*Me*(Me*a-Uinf)/Uinf)*rho0, format='lil')

def rho_getnode(msh):
    '''
    Add rho (external air density) computation functions and return correspondent node,
    to be used with a mesh
    '''

    rhofunc=func(f=rhof, derivs=(drhof_dMe,), args=[0], sparse=True, haspassive=True)

    rho_node=node(f=rhofunc, args_to_inds=['Me'], outs_to_inds=['rho'], passive=msh.passive)

    return rho_node

def Rethf(qe, rho, th11, passive):
    return qe*rho*th11/passive['atm'].mu

def dRethf_dqe(qe, rho, th11, passive):
    return sps.diags(rho*th11/passive['atm'].mu, format='lil')

def dRethf_drho(qe, rho, th11, passive):
    return sps.diags(qe*th11/passive['atm'].mu, format='lil')

def dRethf_dth11(qe, rho, th11, passive):
    return sps.diags(qe*rho/passive['atm'].mu, format='lil')

def Reth_getnode(msh):
    '''
    Add Reth (Reynolds number in respect to momentum thickness) computation functions and return correspondent node,
    for a given mesh
    '''

    Rethfunc=func(f=Rethf, derivs=(dRethf_dqe, dRethf_drho, dRethf_dth11,), args=[0, 1, 2], sparse=True, haspassive=True)

    Rethnode=node(f=Rethfunc, args_to_inds=['qe', 'rho', 'th11'], outs_to_inds=['Reth'], passive=msh.passive)

    return Rethnode
