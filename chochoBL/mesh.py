import numpy as np
import numpy.linalg as lg
import scipy.sparse as sps
import fluids.atmosphere as atm
import time as tm

from residual import *
from three_equation import three_equation as three
from three_equation_b import three_equation_b as teb

_argord=['n', 'th11', 'h', 'beta', 'nts', 'qx', 'qy', 'qz']
_outord=['rmass', 'rmomx', 'rmomz', 'ren', 'rts']

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

def dcell_dnode_Jacobian(vset, correspondence, nnodes):
    '''
    Given a set of vectors corresponding to a set of nodes and an array of correspondences (shape (nnodes, 4))
    between cell and node indexes, returns and the Jacobian of their combination 4 vectors (corresponding to 
    the set of indexes 1, 2, 3, and 4 in the cells).
    Argument correspondence corresponds to a shape (ncells, 4) array with cell.indset indexes
    The Jacobian is returned with respect to the stacked vectors, not to any mixing.
    Example:
    given (a, b, c), returns:
    {a11, a12, a13, a14, b11, b12, b13, b14, ..., cnm}, J
    '''

    nprop=len(vset)
    nv=nnodes
    ncells=np.size(correspondence, axis=0)

    rows=[]
    cols=[]

    for i in range(4):
        for j in range(nprop):
            for k in range(ncells):
                rows.append(k*4*nprop+4*j+i)
                cols.append(correspondence[k, i]+j*nv)
    
    rows=np.array(rows)
    cols=np.array(cols)

    data=np.ones_like(rows)
    
    return sps.coo_matrix((data, (rows, cols)), shape=(4*ncells*nprop, nv*nprop))

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

        self.dcell_dnode={}

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

        self.cell_Mtosys=np.array(self.cell_Mtosys, order='F')

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

        self.rmass, self.rmomx, self.rmomz, self.ren, self.rts=three.mesh_getresiduals(self.cellmatrix, n, th11, h, beta, nts, qx, qy, qz, \
            atm.rho, atm.v_sonic, self.passive['A_transition'], self.passive['A_Rethcrit'], \
                self.cell_Mtosys, self.passive['Uinf'], atm.mu, self.passive['Ncrit'], self.passive['gamma'], \
                    self.rvj, self.rdxj, self.rdyj, self.rudxj, self.rudyj)

        return self.rmass, self.rmomx, self.rmomz, self.ren, self.rts

    def mesh_getresiduals_b(self, n, th11, h, beta, nts, qx, qy, qz,
        rmass_b=None, rmomx_b=None, rmomz_b=None, ren_b=None, rts_b=None):
        '''
        Run reverse AD code for mesh residual module.
        Seeds set as None in kwargs will be interpreted as zero
        '''

        nb=np.zeros(self.nnodes)
        th11b=np.zeros(self.nnodes)
        hb=np.zeros(self.nnodes)
        betab=np.zeros(self.nnodes)
        ntsb=np.zeros(self.nnodes)
        qxb=np.zeros(self.nnodes)
        qyb=np.zeros(self.nnodes)
        qzb=np.zeros(self.nnodes)

        atm=self.passive['atm']

        teb.mesh_getresiduals_b(
            self.cellmatrix, 
                n, nb, 
                    th11, th11b, 
                        h, hb, 
                            beta, betab, 
                                nts, ntsb, 
                                    qx, qxb, qy, qyb, qz, qzb,
                                        atm.rho, atm.v_sonic, 
                                            self.passive['A_transition'], self.passive['A_Rethcrit'], 
                                                self.cell_Mtosys, 
                                                    self.passive['Uinf'], 
                                                        atm.mu,
                                                            self.passive['Ncrit'], 
                                                                self.passive['gamma'], 
                                                                    self.rvj, self.rdxj, self.rdyj, self.rudxj, self.rudyj, 
                                                                        self.rmass, (rmass_b if rmass_b is not None else np.zeros_like(self.rmass)), 
                                                                            self.rmomx, (rmomx_b if rmomx_b is not None else np.zeros_like(self.rmomx)), 
                                                                                self.rmomz, (rmomz_b if rmomz_b is not None else np.zeros_like(self.rmomz)), 
                                                                                    self.ren, (ren_b if ren_b is not None else np.zeros_like(self.ren)), 
                                                                                        self.rts, (rts_b if rts_b is not None else np.zeros_like(self.rts)))

        return nb, th11b, hb, betab, ntsb, qxb, qyb, qzb

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
        
        self.dcell_dnode[nv]=dcell_dnode_Jacobian(vset, self.cellmatrix-1, self.nnodes)

        return self.dcell_dnode[nv]

    def _Jac_liladapt(self, J, i):
        '''
        Adapt a Jacobian from output format of three_equation_b.mesh_getresiduals_jac
        to lil matrix Jacobian of nodal residuals
        '''

        distJ=self.dcell_dnode_compose((1.0,))

        ncells=self.ncells

        rows=[]
        cols=[]
        data=[]

        for nc in range(ncells):
            for nm in range(4):
                for nn in range(4):
                    rows.append(nc+nm)
                    cols.append(nc+nn)
                    data.append(J[nc, nm, nn, i])
        
        rows=np.array(rows)
        cols=np.array(cols)
        data=np.array(data)

        return distJ.T@sps.coo_matrix((data, (rows, cols)), shape=(4*ncells, 4*ncells))@distJ

    def _Jac_liladapt_dict(self, J):
        '''
        Adapt a Jacobian from output format of three_equation_b.mesh_getresiduals_jac
        to lil matrix Jacobian of nodal residuals. Return for all arguments,
        in dictionary
        '''

        return {p:self._Jac_liladapt(J, i) for i, p in enumerate(_argord)}
    
    def mesh_getresiduals_jac(self, n, th11, h, beta, nts, qx, qy, qz):
        '''
        Get Jacobian of residuals in respect to input variables
        '''

        atm=self.passive['atm']

        rmassj, rmomxj, rmomzj, renj, rtsj=teb.mesh_getresiduals_jac(self.cellmatrix, n, th11, \
           h, beta, nts, qx, qy, qz, \
               atm.rho, atm.v_sonic, self.passive['A_transition'], self.passive['A_Rethcrit'], self.cell_Mtosys, self.passive['Uinf'], atm.mu, self.passive['Ncrit'], \
                   self.passive['gamma'], self.rvj, self.rdxj, self.rdyj, self.rudxj, self.rudyj, self.rmass, self.rmomx, \
                       self.rmomz, self.ren, self.rts)

        return {
            'rmass':self._Jac_liladapt_dict(rmassj),
            'rmomx':self._Jac_liladapt_dict(rmomxj),
            'rmomz':self._Jac_liladapt_dict(rmomzj),
            'ren':self._Jac_liladapt_dict(renj),
            'rts':self._Jac_liladapt_dict(rtsj)
        }

    def mesh_solveJac(self):
        '''
        Solve problem [J](Dx)=-(r) and return Dx
        '''

        nCCs=np.setdiff1d(np.arange(self.nnodes, dtype='int'), self.CC)
