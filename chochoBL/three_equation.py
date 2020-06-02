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

    def uwfs(qx, qy, qz, passive):
        value={'u':uf(qx, qy, qz, passive), 'w':wf(qx, qy, qz, passive)}
        Jac={'u':{'qx':msh.Juqx, 'qy':msh.Juqy, 'qz':msh.Juqz}, 'w':{'qx':msh.Jwqx, 'qy':msh.Jwqy, 'qz':msh.Jwqz}}

        return value, Jac

    uwnode=node(f=uwfs, args_to_inds=['qx', 'qy', 'qz'], outs_to_inds=['u', 'w'], haspassive=True, passive=msh.passive)

    return uwnode

def qef(qx, qy, qz):
    return np.sqrt(qx**2+qy**2+qz**2)

def dqef_dqx(qx, qy, qz, qe):
    return sps.diags(qx/qe, format='lil')

def dqef_dqy(qx, qy, qz, qe):
    return sps.diags(qy/qe, format='lil')

def dqef_dqz(qx, qy, qz, qe):
    return sps.diags(qz/qe, format='lil')

def qe_getnode(msh):
    '''
    Add qe (scalar) computation functions and return correspondent node, to be used with a mesh
    '''

    def qefunc(qx, qy, qz):
        qe=qef(qx, qy, qz)

        value={'qe':qe}
        Jac={'qe':{'qx':dqef_dqx(qx, qy, qz, qe), 'qy':dqef_dqy(qx, qy, qz, qe), 'qz':dqef_dqz(qx, qy, qz, qe)}}

        return value, Jac

    qenode=node(f=qefunc, args_to_inds=['qx', 'qy', 'qz'], outs_to_inds=['qe'], passive=msh.passive)

    return qenode

def Mef(qe, passive):
    return qe/passive['atm'].v_sonic

def dMef_dqe(qe, passive):
    return sps.eye(len(qe), format='lil')/passive['atm'].v_sonic

def Me_getnode(msh):
    '''
    Add Me (external Mach number) computation functions and return correspondent node,
    to be used with a mesh
    '''

    def Mefunc(qe, passive):
        value={'Me':Mef(qe, passive)}
        Jac={'Me':{'qe':dMef_dqe(qe, passive)}}

        return value, Jac

    Menode=node(f=Mefunc, args_to_inds=['qe'], outs_to_inds=['Me'], haspassive=True, passive=msh.passive)

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

    def rhofunc(Me, passive):
        value={'rho':rhof(Me, passive)}
        Jac={'rho':{'Me':drhof_dMe(Me, passive)}}

        return value, Jac

    rho_node=node(f=rhofunc, args_to_inds=['Me'], outs_to_inds=['rho'], passive=msh.passive, haspassive=True)

    return rho_node

def Rethf(qe, rho, th11, passive):
    return qe*rho*th11/passive['atm'].mu

def dRethf_dqe(qe, Reth, passive):
    return sps.diags(Reth/qe, format='lil')

def dRethf_drho(rho, Reth, passive):
    return sps.diags(Reth/rho, format='lil')

def dRethf_dth11(th11, Reth, passive):
    return sps.diags(Reth/th11, format='lil')

def Reth_getnode(msh):
    '''
    Add Reth (Reynolds number in respect to momentum thickness) computation functions and return correspondent node,
    for a given mesh
    '''

    def Rethfunc(qe, rho, th11, passive):
        Reth=Rethf(qe, rho, th11, passive)

        value={'Reth':Reth}
        Jac={'Reth':{'qe':dRethf_dqe(qe, Reth, passive), 'rho':dRethf_drho(rho, Reth, passive), 'th11':dRethf_dth11(th11, Reth, passive)}}

        return value, Jac

    Rethnode=node(f=Rethfunc, args_to_inds=['qe', 'rho', 'th11'], outs_to_inds=['Reth'], passive=msh.passive, haspassive=True)

    return Rethnode

def _innode_linemult(strip, V):
    return sum([m*v for v, m in zip(V, strip)])

def _innode_matmul(M, V, q=None):
    return tuple(_innode_linemult(strip, V) if q is None else _innode_linemult(strip, V)*q for strip in M)

import time as tm

def J_innode(th11, th12, th21, th22, u, w, rho, passive):
    Jac={
        'Jxx':{
            'u':None,
            'w':None
        },
        'Jxz':{
            'u':None,
            'w':None
        },
        'Jzx':{
            'u':None,
            'w':None
        },
        'Jzz':{
            'u':None,
            'w':None
        }
    }

    msh=passive['mesh']

    distJ=msh.dcell_dnode_compose((th11,))

    indexing=diag_cell_indexing(msh.cellmatrix, msh.nnodes, msh.ncells)

    th11_c=distJ@th11
    th12_c=distJ@th12
    th21_c=distJ@th21
    th22_c=distJ@th22

    rho_c=distJ@rho

    th_c=[th11_c, th12_c, th21_c, th22_c]

    vels=[
        [u**2, -u*w, -u*w, w**2],
        [u*w, u**2, -w**2, -u*w],
        [u*w, -w**2, u**2, -u*w],
        [w**2, u*w, u*w, u**2]
    ]

    dvels_du=[
        [2*u, -w, -w, 0.0],
        [w, 2*u, 0.0, -w],
        [w, 0.0, 2*u, -w],
        [0.0, w, w, 2*u]
    ]

    dvels_dw=[
        [0.0, -u, -u, 2*w],
        [u, 0.0, -2*w, -u],
        [u, -2*w, 0.0, -u],
        [2*w, u, u, 0.0]
    ]

    #start without multiplying by rho
    Jxx, Jxz, Jzx, Jzz=_innode_matmul(vels, th_c)

    Jac['Jxx']['u'], Jac['Jxz']['u'], Jac['Jzx']['u'], Jac['Jzz']['u']=_innode_matmul(dvels_du, th_c, q=rho_c)
    Jac['Jxx']['w'], Jac['Jxz']['w'], Jac['Jzx']['w'], Jac['Jzz']['w']=_innode_matmul(dvels_dw, th_c, q=rho_c)

    Jac['Jxx']['rho']=diag_cell_Jacobian(Jxx, indexing)
    Jac['Jxz']['rho']=diag_cell_Jacobian(Jxz, indexing)
    Jac['Jzx']['rho']=diag_cell_Jacobian(Jzx, indexing)
    Jac['Jzz']['rho']=diag_cell_Jacobian(Jzz, indexing)

    Jxx, Jxz, Jzx, Jzz=_innode_matmul(vels, th_c, q=rho_c)

    value={'Jxx':Jxx, 'Jxz':Jxz, 'Jzx':Jzx, 'Jzz':Jzz}

    Jac['Jxx'].update(
        {
            'th11':diag_cell_Jacobian(vels[0][0]*rho_c, indexing),
            'th12':diag_cell_Jacobian(vels[0][1]*rho_c, indexing),
            'th21':diag_cell_Jacobian(vels[0][2]*rho_c, indexing),
            'th22':diag_cell_Jacobian(vels[0][3]*rho_c, indexing)
        }
    )

    Jac['Jxz'].update(
        {
            'th11':diag_cell_Jacobian(vels[1][0]*rho_c, indexing),
            'th12':diag_cell_Jacobian(vels[1][1]*rho_c, indexing),
            'th21':diag_cell_Jacobian(vels[1][2]*rho_c, indexing),
            'th22':diag_cell_Jacobian(vels[1][3]*rho_c, indexing)
        }
    )

    Jac['Jzx'].update(
        {
            'th11':diag_cell_Jacobian(vels[2][0]*rho_c, indexing),
            'th12':diag_cell_Jacobian(vels[2][1]*rho_c, indexing),
            'th21':diag_cell_Jacobian(vels[2][2]*rho_c, indexing),
            'th22':diag_cell_Jacobian(vels[2][3]*rho_c, indexing)
        }
    )

    Jac['Jzz'].update(
        {
            'th11':diag_cell_Jacobian(vels[3][0]*rho_c, indexing),
            'th12':diag_cell_Jacobian(vels[3][1]*rho_c, indexing),
            'th21':diag_cell_Jacobian(vels[3][2]*rho_c, indexing),
            'th22':diag_cell_Jacobian(vels[3][3]*rho_c, indexing)
        }
    )
    
    return value, Jac

def J_getnode(msh):
    '''
    Return node for calculation of J momentum transport tensor
    '''

    Jnode=node(f=J_innode, args_to_inds=['th11', 'th12', 'th21', 'th22', 'u', 'w', 'rho'], outs_to_inds=['Jxx', 'Jxz', 'Jzx', 'Jzz'], \
        passive=msh.passive, haspassive=True)

    return Jnode

def M_innode(deltastar_1, deltastar_2, u, w, rho, passive):
    msh=passive['mesh']

    distJ=msh.dcell_dnode_compose((deltastar_1,))

    rho_c=distJ@rho

    deltastar_1_c=distJ@deltastar_1
    deltastar_2_c=distJ@deltastar_2

    indexing=diag_cell_indexing(msh.cellmatrix, msh.nnodes, msh.ncells)

    rhou_c=rho_c*u
    rhow_c=rho_c*w

    Mx=deltastar_1_c*rhou_c+deltastar_2_c*rhow_c
    Mz=deltastar_2_c*rhou_c-deltastar_1_c*rhow_c

    value={
        'Mx':Mx,
        'Mz':Mz
    }

    Jac={
        'Mx':{
            'deltastar_1':diag_cell_Jacobian(rhou_c, indexing),
            'deltastar_2':diag_cell_Jacobian(rhow_c, indexing),
            'u':deltastar_1_c*rho_c,
            'w':deltastar_2_c*rho_c,
            'rho':diag_cell_Jacobian(Mx/rho_c, indexing)
        },
        'Mz':{
            'deltastar_1':diag_cell_Jacobian(-rhow_c, indexing),
            'deltastar_2':diag_cell_Jacobian(rhou_c, indexing),
            'u':deltastar_2_c*rho_c,
            'w':-deltastar_1_c*rho_c,
            'rho':diag_cell_Jacobian(Mz/rho_c, indexing)
        }
    }

    return value, Jac

def M_getnode(msh):
    '''
    Return a node for calculating mass flux vector
    '''

    Mnode=node(f=M_innode, args_to_inds=['deltastar_1', 'deltastar_2', 'u', 'w', 'rho'], outs_to_inds=['Mx', 'Mz'], passive=msh.passive, haspassive=True)

    return Mnode

def E_innode(thetastar_1, thetastar_2, u, w, qe, rho, passive):
    msh=passive['mesh']

    distJ=msh.dcell_dnode_compose((thetastar_1,))

    rho_c=distJ@rho
    qe_c=distJ@qe

    rhoq2_c=rho_c*qe_c**2

    rhouq2_c=rhoq2_c*u
    rhowq2_c=rhoq2_c*w

    thetastar_1_c=distJ@thetastar_1
    thetastar_2_c=distJ@thetastar_2

    indexing=diag_cell_indexing(msh.cellmatrix, msh.nnodes, msh.ncells)

    Ex=rhouq2_c*thetastar_1_c+rhowq2_c*thetastar_2_c
    Ez=rhouq2_c*thetastar_2_c-rhowq2_c*thetastar_1_c

    value={
        'Ex':Ex,
        'Ez':Ez
    }

    Jac={
        'Ex':{
            'thetastar_1':diag_cell_Jacobian(rhouq2_c, indexing),
            'thetastar_2':diag_cell_Jacobian(rhowq2_c, indexing),
            'u':rhoq2_c*thetastar_1_c,
            'w':rhoq2_c*thetastar_2_c,
            'qe':diag_cell_Jacobian(2*Ex/qe_c, indexing),
            'rho':diag_cell_Jacobian(Ex/rho_c, indexing)
        },
        'Ez':{
            'thetastar_1':diag_cell_Jacobian(-rhowq2_c, indexing),
            'thetastar_2':diag_cell_Jacobian(rhouq2_c, indexing),
            'u':rhoq2_c*thetastar_2_c,
            'w':-rhoq2_c*thetastar_1_c,
            'qe':diag_cell_Jacobian(2*Ez/qe_c, indexing),
            'rho':diag_cell_Jacobian(Ez/rho_c, indexing)
        }
    }

    return value, Jac

def E_getnode(msh):
    '''
    Returns a node for calculation of energy dissipation vector
    '''

    Enode=node(f=E_innode, args_to_inds=['thetastar_1', 'thetastar_2', 'u', 'w', 'qe', 'rho'], outs_to_inds=['Ex', 'Ez'], passive=msh.passive, haspassive=True)

    return Enode

def rhoQ_innode(deltaprime_1, deltaprime_2, u, w, rho, passive):
    msh=passive['mesh']

    distJ=msh.dcell_dnode_compose((deltaprime_1,))

    rho_c=distJ@rho

    deltaprime_1_c=distJ@deltaprime_1
    deltaprime_2_c=distJ@deltaprime_2

    indexing=diag_cell_indexing(msh.cellmatrix, msh.nnodes, msh.ncells)

    rhou_c=rho_c*u
    rhow_c=rho_c*w

    rhoQx=deltaprime_1_c*rhou_c+deltaprime_2_c*rhow_c
    rhoQz=deltaprime_2_c*rhou_c-deltaprime_1_c*rhow_c

    value={
        'rhoQx':rhoQx,
        'rhoQz':rhoQz
    }

    Jac={
        'rhoQx':{
            'deltaprime_1':diag_cell_Jacobian(rhou_c, indexing),
            'deltaprime_2':diag_cell_Jacobian(rhow_c, indexing),
            'u':deltaprime_1_c*rho_c,
            'w':deltaprime_2_c*rho_c,
            'rho':diag_cell_Jacobian(rhoQx/rho_c, indexing)
        },
        'rhoQz':{
            'deltaprime_1':diag_cell_Jacobian(-rhow_c, indexing),
            'deltaprime_2':diag_cell_Jacobian(rhou_c, indexing),
            'u':deltaprime_2_c*rho_c,
            'w':-deltaprime_1_c*rho_c,
            'rho':diag_cell_Jacobian(rhoQz/rho_c, indexing)
        }
    }

    return value, Jac

def rhoQ_getnode(msh):
    '''
    Return a node for calculating mass flux vector
    '''

    rhoQnode=node(f=rhoQ_innode, args_to_inds=['deltaprime_1', 'deltaprime_2', 'u', 'w', 'rho'], outs_to_inds=['rhoQx', 'rhoQz'], passive=msh.passive, haspassive=True)

    return rhoQnode

def D_innode(Cd_1, Cd_2, rho, qe, passive):
    msh=passive['mesh']

    distJ=msh.dcell_dnode_compose((Cd_1,))

    indexing=diag_cell_indexing(msh.cellmatrix, msh.nnodes, msh.ncells)

    CD=distJ@(Cd_1+Cd_2)
    qe_c=distJ@qe
    rho_c=distJ@rho

    dD_dCD=qe_c**3*rho_c
    dD_dq=3*qe_c**2*rho_c*CD
    dD_drho=qe_c**3*CD

    D=dD_dCD*CD

    dD_dCd_1=dD_dCD
    dD_dCd_2=dD_dCD

    value={'D':D}

    Jac={
        'D':{
            'Cd':diag_cell_Jacobian(dD_dCd_1, indexing),
            'Cd_2':diag_cell_Jacobian(dD_dCd_2, indexing),
            'rho':diag_cell_Jacobian(dD_drho, indexing),
            'qe':diag_cell_Jacobian(dD_dq, indexing)
        }
    }

    return value, Jac

def D_getnode(msh):
    '''
    Return a node to calculate dissipation
    '''

    D_node=node(f=D_innode, args_to_inds=['Cd', 'Cd_2', 'rho', 'qe'], outs_to_inds=['D'], passive=msh.passive, haspassive=True)

    return D_node