import numpy as np
import numpy.linalg as lg
import time as tm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import fluids.atmosphere as atm

from garlekin import *
from differentiation import *
from three_equation import *
from closure import *
from transition import *

def _gen_orthonormal(u, n):
    '''
    Create an orthonormal coordinate system returning the Universal-to-Local coordinate conversion matrix.
    Based on x or z versor and local normal vector. y direction is normal
    '''

    M=np.zeros((3, 3))

    M[0, :]=u/lg.norm(u)
    
    M[1, :]=n-M[0, :]*(n@M[0, :])
    M[1, :]/=lg.norm(M[1, :])

    M[2, :]=np.cross(M[0, :], M[1, :])

    return M

class cell:
    '''
    Class containing info about a single Garlekin cell
    '''

    def __init__(self, indset):
        '''
        Instantiate a cell with a given set of node indexes
        '''
        
        self.indset=np.array(list(indset))
    
    def local_system_build(self, coords, normals):
        '''
        Construct local normal coordinate system based on center (c variable in the class)
        and local coordinate system (orthonormalized). Normal vector is deduced from mean
        of "normals" argument line

        ====
        args
        ====
        coords: shape (4, 3) array
        normals: shape (4, 3) array
        '''

        c=np.mean(coords, axis=0)

        n=np.mean(normals, axis=0)
        n/=lg.norm(n)

        self.c=c
        
        if 1.0-n[0]**2<1e-4:
            self.Mtosys=_gen_orthonormal(np.array([0.0, 0.0, 1.0]), n)
        else:
            self.Mtosys=_gen_orthonormal(np.array([1.0, 0.0, 0.0]), n)

        coords[:, 0]-=c[0]
        coords[:, 1]-=c[1]
        coords[:, 2]-=c[2]

        coords=coords@self.Mtosys.T

        self.xs=coords[:, 0]
        self.zs=coords[:, 2]

        order=np.argsort(np.arctan2(self.zs, self.xs))

        self.indset=self.indset[order]
        self.xs=self.xs[order]
        self.zs=self.zs[order]

        self.Rv=v_residual_matrix(self.xs, self.zs)
        self.Rdvdx=dvdx_residual_matrix(self.xs, self.zs)
        self.Rdvdz=dvdy_residual_matrix(self.xs, self.zs)
        self.Rudvdx=udvdx_residual_matrix(self.xs, self.zs)
        self.Rudvdz=udvdy_residual_matrix(self.xs, self.zs)

defatm=atm.ATMOSPHERE_1976(0.0)

class mesh:
    '''
    Class containing information about a mesh
    '''

    def __init__(self, atm_props=defatm, Uinf=1.0, Ncrit=6.0, A_transition=50.0, A_Rethcrit=1.0, gamma=1.4):
        '''
        Initialize a mesh object without any nodes or cells
        '''

        self.nodes=[]
        self.cells=[]
        self.node_Mtosys=[]

        self.passive={'mesh':self, 'atm':atm_props, 'Uinf':Uinf, 'Ncrit':Ncrit, 'A_transition':A_transition, \
            'A_Rethcrit':A_Rethcrit, 'gamma':gamma}

        self.dcell_dnode={}
    
    def add_node(self, coords):
        '''
        Add a new node to list based on coordinates.
        '''

        c=list(coords)
        if not c in self.nodes:
            self.nodes.append(c)
        
        return len(self.nodes)-1
    
    def add_cell(self, indset):
        '''
        Add cell indexes to list before assembling
        '''

        add_set(self.cells, indset)
    
    def compose(self, normals):
        '''
        Compose local coordinate systems and store local coordinates
        '''

        #turn nodes into an array
        self.nodes=np.array(self.nodes)

        #converting cells to class structure
        newcells=[]

        for i, inds in enumerate(self.cells):
            newcells.append(cell(inds))

            #develop local coordinate system and store xs and ys
            newcells[-1].local_system_build(self.nodes[newcells[-1].indset, :], normals[newcells[-1].indset, :])
        
        self.cells=newcells
        self.cellmatrix=np.array([list(c.indset) for c in newcells], dtype='int')

        self.nnodes=len(self.nodes)
        self.ncells=len(self.cells)

    def dcell_dnode_compose(self, vset):
        '''
        Compose the dcell_dnode Jacobian for the matrix with a given number of mixed vectors,
        represented in tuple vset. It os only created if not yet present for the given number of
        vectors.
        
        Returns the necessary Jacobian
        '''

        nv=len(vset)

        if nv in self.dcell_dnode:
            return self.dcell_dnode[nv]
        
        self.dcell_dnode[nv]=dcell_dnode_Jacobian(vset, self.cellmatrix)

        return self.dcell_dnode[nv]

    def graph_init(self):
        '''
        Initialize the generation of a graph for algorithmic differentiation
        '''

        self.gr=graph()

        #setting heads
        q_head=head(outs_to_inds=['qx', 'qy', 'qz'], passive=self.passive)
        theta11_head=head(outs_to_inds=['th11'], passive=self.passive)
        H_head=head(outs_to_inds=['H'], passive=self.passive)
        N_head=head(outs_to_inds=['N'], passive=self.passive)
        beta_head=head(outs_to_inds=['beta'], passive=self.passive)

        #adding nodes
        uw_node=uw_conversion_getnode(self)
        qe_node=qe_getnode(self)
        Me_node=Me_getnode(self)
        rho_node=rho_getnode(self)
        Reth_node=Reth_getnode(self)
        Hk_node=Hk_getnode(self)
        p_node=p_getnode(self)
        sigma_N_node=sigma_N_getnode(self)
        closure_node=closure_getnode(self)
        A_node=A_getnode(self)
        deltastar_node=deltastar_getnode(self)
        Cf_node=Cf_getnode(self)
        theta_node=theta_getnode(self)
        thetastar_node=thetastar_getnode(self)
        deltaprime_node=deltaprime_getnode(self)
        Cd_node=Cd_getnode(self)
        J_node=J_getnode(self)

        #adding nodes
        self.gr.add_node(q_head, 'q', head=True)
        self.gr.add_node(theta11_head, 'th11', head=True)
        self.gr.add_node(H_head, 'H', head=True)
        self.gr.add_node(N_head, 'N', head=True)
        self.gr.add_node(beta_head, 'beta', head=True)

        self.gr.add_node(uw_node, 'uw')
        self.gr.add_node(qe_node, 'qe')
        self.gr.add_node(Me_node, 'Me')
        self.gr.add_node(rho_node, 'rho')
        self.gr.add_node(Reth_node, 'Reth')
        self.gr.add_node(Hk_node, 'Hk')
        self.gr.add_node(p_node, 'p')
        self.gr.add_node(sigma_N_node, 'sigma_N')
        self.gr.add_node(closure_node, 'closure')
        self.gr.add_node(A_node, 'A')
        self.gr.add_node(deltastar_node, 'deltastar')
        self.gr.add_node(Cf_node, 'Cf')
        self.gr.add_node(theta_node, 'theta')
        self.gr.add_node(thetastar_node, 'thetastar')
        self.gr.add_node(deltaprime_node, 'deltaprime')
        self.gr.add_node(Cd_node, 'Cd')
        self.gr.add_node(J_node, 'J')

        #adding edges
        e_q_uw=edge(q_head, uw_node, {'qx', 'qy', 'qz'})
        e_q_qe=edge(q_head, qe_node, {'qx', 'qy', 'qz'})
        e_q_Me=edge(qe_node, Me_node, {'qe'})
        e_Me_rho=edge(Me_node, rho_node, {'Me'})
        e_qe_Reth=edge(qe_node, Reth_node, {'qe'})
        e_rho_Reth=edge(rho_node, Reth_node, {'rho'})
        e_th11_Reth=edge(theta11_head, Reth_node, {'th11'})
        e_H_Hk=edge(H_head, Hk_node, {'H'})
        e_Me_Hk=edge(Me_node, Hk_node, {'Me'})
        e_Hk_p=edge(Hk_node, p_node, {'Hk'})
        e_Reth_p=edge(Reth_node, p_node, {'Reth'})
        e_th11_p=edge(theta11_head, p_node, {'th11'})
        e_N_sigma_N=edge(N_head, sigma_N_node, {'N'})
        e_sigma_N_closure=edge(sigma_N_node, closure_node, {'sigma_N'})
        e_Reth_closure=edge(Reth_node, closure_node, {'Reth'})
        e_Me_closure=edge(Me_node, closure_node, {'Me'})
        e_Hk_closure=edge(Hk_node, closure_node, {'Hk'})
        e_closure_A=edge(closure_node, A_node, {'Cf'})
        e_Me_A=edge(Me_node, A_node, {'Me'})
        e_beta_A=edge(beta_head, A_node, {'beta'})
        e_H_deltastar=edge(H_head, deltastar_node, {'H'})
        e_th11_deltastar=edge(theta11_head, deltastar_node, {'th11'})
        e_A_deltastar=edge(A_node, deltastar_node, {'A'})
        e_A_Cf=edge(A_node, Cf_node, {'tanb'})
        e_closure_Cf=edge(closure_node, Cf_node, {'Cf'})
        e_th11_theta=edge(theta11_head, theta_node, {'th11'})
        e_deltastar_theta=edge(deltastar_node, theta_node, {'deltastar_2'})
        e_A_theta=edge(A_node, theta_node, {'A'})
        e_closure_thetastar=edge(closure_node, thetastar_node, {'Hstar'})
        e_deltastar_thetastar=edge(deltastar_node, thetastar_node, {'deltastar_1'})
        e_theta_thetastar=edge(theta_node, thetastar_node, {'th22'})
        e_th11_thetastar=edge(theta11_head, thetastar_node, {'th11'})
        e_A_thetastar=edge(A_node, thetastar_node, {'A'})
        e_closure_deltaprime=edge(closure_node, deltaprime_node, {'Hprime'})
        e_A_deltaprime=edge(A_node, deltaprime_node, {'A'})
        e_th11_deltaprime=edge(theta11_head, deltaprime_node, {'th11'})
        e_closure_Cd=edge(closure_node, Cd_node, {'Cd'})
        e_A_Cd=edge(A_node, Cd_node, {'A'})
        e_th11_J=edge(theta11_head, J_node, {'th11'})
        e_theta_J=edge(theta_node, J_node, {'th12', 'th21', 'th22'})
        e_uw_J=edge(uw_node, J_node, {'u', 'w'})
        e_rho_J=edge(rho_node, J_node, {'rho'})

        self.gr.add_edge(e_q_uw)
        self.gr.add_edge(e_q_qe)
        self.gr.add_edge(e_q_Me)
        self.gr.add_edge(e_Me_rho)
        self.gr.add_edge(e_qe_Reth)
        self.gr.add_edge(e_rho_Reth)
        self.gr.add_edge(e_th11_Reth)
        self.gr.add_edge(e_H_Hk)
        self.gr.add_edge(e_Me_Hk)
        self.gr.add_edge(e_Hk_p)
        self.gr.add_edge(e_Reth_p)
        self.gr.add_edge(e_th11_p)
        self.gr.add_edge(e_N_sigma_N)
        self.gr.add_edge(e_sigma_N_closure)
        self.gr.add_edge(e_Reth_closure)
        self.gr.add_edge(e_Me_closure)
        self.gr.add_edge(e_Hk_closure)
        self.gr.add_edge(e_closure_A)
        self.gr.add_edge(e_Me_A)
        self.gr.add_edge(e_beta_A)
        self.gr.add_edge(e_H_deltastar)
        self.gr.add_edge(e_th11_deltastar)
        self.gr.add_edge(e_A_deltastar)
        self.gr.add_edge(e_A_Cf)
        self.gr.add_edge(e_closure_Cf)
        self.gr.add_edge(e_th11_theta)
        self.gr.add_edge(e_deltastar_theta)
        self.gr.add_edge(e_A_theta)
        self.gr.add_edge(e_closure_thetastar)
        self.gr.add_edge(e_A_thetastar)
        self.gr.add_edge(e_th11_thetastar)
        self.gr.add_edge(e_theta_thetastar)
        self.gr.add_edge(e_deltastar_thetastar)
        self.gr.add_edge(e_closure_deltaprime)
        self.gr.add_edge(e_A_deltaprime)
        self.gr.add_edge(e_th11_deltaprime)
        self.gr.add_edge(e_closure_Cd)
        self.gr.add_edge(e_A_Cd)
        self.gr.add_edge(e_th11_J)
        self.gr.add_edge(e_theta_J)
        self.gr.add_edge(e_uw_J)
        self.gr.add_edge(e_rho_J)
