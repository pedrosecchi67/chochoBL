import numpy as np
import numpy.linalg as lg
import time as tm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import fluids.atmosphere as atm

from garlekin import *
from differentiation import *
from three_equation import *

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

    def __init__(self, atm_props=defatm, Uinf=1.0):
        '''
        Initialize a mesh object without any nodes or cells
        '''

        self.nodes=[]
        self.cells=[]
        self.node_Mtosys=[]

        self.passive={'mesh':self, 'atm':atm_props, 'Uinf':Uinf}
    
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

    def graph_init(self):
        '''
        Initialize the generation of a graph for algorithmic differentiation
        '''

        self.gr=graph()

        #setting heads
        q_head=head(outs_to_inds=['qx', 'qy', 'qz'], passive=self.passive)
        theta11_head=head(outs_to_inds=['th11'], passive=self.passive)
        H_head=head(outs_to_inds=['H'], passive=self.passive)

        #adding nodes
        uw_node=uw_conversion_getnode(self)
        qe_node=qe_getnode(self)
        Me_node=Me_getnode(self)
        rho_node=rho_getnode(self)
        Reth_node=Reth_getnode(self)

        #adding nodes
        self.gr.add_node(q_head, 'q', head=True)
        self.gr.add_node(theta11_head, 'th11', head=True)
        self.gr.add_node(H_head, 'H', head=True)

        self.gr.add_node(uw_node, 'uw')
        self.gr.add_node(qe_node, 'qe')
        self.gr.add_node(Me_node, 'Me')
        self.gr.add_node(rho_node, 'rho')
        self.gr.add_node(Reth_node, 'Reth')

        #adding edges
        e_q_uw=edge(q_head, uw_node, {'qx', 'qy', 'qz'})
        e_q_qe=edge(q_head, qe_node, {'qx', 'qy', 'qz'})
        e_q_Me=edge(qe_node, Me_node, {'qe'})
        e_Me_rho=edge(Me_node, rho_node, {'Me'})
        e_qe_Reth=edge(qe_node, Reth_node, {'qe'})
        e_rho_Reth=edge(rho_node, Reth_node, {'rho'})
        e_th11_Reth=edge(theta11_head, Reth_node, {'th11'})

        self.gr.add_edge(e_q_uw)
        self.gr.add_edge(e_q_qe)
        self.gr.add_edge(e_q_Me)
        self.gr.add_edge(e_Me_rho)
        self.gr.add_edge(e_qe_Reth)
        self.gr.add_edge(e_rho_Reth)
        self.gr.add_edge(e_th11_Reth)
