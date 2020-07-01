import numpy as np
import numpy.linalg as lg
import time as tm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import fluids.atmosphere as atm

from residual import *

def add_set(l, s):
    '''
    Add set s to list l if not yet present
    '''

    if not s in l:
        l.append(s)
        return len(l)-1
    else:
        return l.index(s)

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

        self.cell_Mtosys=[]
        self.cell_rvjs=[]
        self.cell_rdxjs=[]
        self.cell_rdyjs=[]
        self.cell_rudxjs=[]
        self.cell_rudyjs=[]

        self.passive={'mesh':self, 'atm':atm_props, 'Uinf':Uinf, 'Ncrit':Ncrit, 'A_transition':A_transition, \
            'A_Rethcrit':A_Rethcrit, 'gamma':gamma}

    def add_node(self, coords):
        '''
        Add a new node to list based on coordinates.
        '''

        c=list(coords)
        if not c in self.nodes:
            self.nodes.append(c)
    
    def add_cell(self, indset):
        '''
        Add cell indexes to list before assembling
        '''

        add_set(self.cells, indset)
    
    def compose(self, normals, transition_CC=None):
        '''
        Compose local coordinate systems and store local coordinates
        '''

        #turn nodes into an array
        self.nodes=np.array(self.nodes)

        self.maxdim=np.amax(self.nodes)

        #converting cells to class structure
        newcells=[]

        for inds in self.cells:
            newcells.append(cell(inds))

            #develop local coordinate system and store xs and ys
            newcells[-1].local_system_build(self.nodes[newcells[-1].indset, :], normals[newcells[-1].indset, :])
        
        self.cells=newcells

        xs=[]
        ys=[]

        for c in self.cells:
            xs.append(c.xs.tolist())
            ys.append(c.zs.tolist())

            self.cell_Mtosys.append(c.Mtosys.tolist())

        self.cell_Mtosys=np.array(self.cell_Mtosys)

        xs=np.array(xs)
        ys=np.array(ys)

        self.rvj, self.rdxj, self.rdyj, self.rudxj, self.rudyj=residual.get_mesh_resmats(xs, ys)

        self.cellmatrix=np.array([list(c.indset) for c in newcells], dtype='int', order='F')+1

        self.nnodes=len(self.nodes)
        self.ncells=len(self.cells)

        self.CC=transition_CC if transition_CC is not None else np.array([], dtype='float')

    def mesh_getresiduals(self, n, th11, h, beta, nts, qx, qy, qz):
        '''
        Calculate residuals for mesh
        '''

        atm=self.passive['atm']

        rmass, rmomx, rmomz, ren, rts=residual.mesh_getresiduals(self.cellmatrix, n, th11, h, beta, nts, qx, qy, qz, \
            atm.rho, atm.v_sonic, self.passive['A_transition'], self.passive['A_Rethcrit'], \
                self.cell_Mtosys, self.passive['Uinf'], atm.mu, self.passive['Ncrit'], self.passive['gamma'], \
                    self.rvj, self.rdxj, self.rdyj, self.rudxj, self.rudyj)

        return rmass, rmomx, rmomz, ren, rts
