import numpy as np
import scipy.sparse as sps

'''
Module with functions and classes to assist in Jacobian handling
and automatic differentiation
'''

def addnone(a, b):
    if a is None:
        if b is None:
            return None
        else:
            return b
    else:
        if b is None:
            return a
        else:
            return a+b

def prod_as_diag(A, B, transpose=True):
    '''
    Return a product of A.T@B, considering one-dimensional arrays considered to be diagonals
    '''

    isdiagA=(A.ndim==1)
    isdiagB=(B.ndim==1)

    if isdiagA:
        if isdiagB:
            return A*B
        else:
            if sps.issparse(B):
                return B.T.multiply(A).T
            else:
                return np.multiply(B.T, A).T
    else:
        if isdiagB:
            if sps.issparse(A):
                return (A.T if transpose else A).multiply(B)
            else:
                return (A.T if transpose else A)*B
        else:
            return (A.T if transpose else A)@B

def _matmulnone(A, B, transpose=True):
    if A is None or B is None:
        return None
    else:
        return prod_as_diag(A, B, transpose=transpose)

class edge:
    '''
    Class defining an edge for an algorithmic differentiation graph.
    '''

    def __init__(self, n1, n2, k):
        '''
        Define an edge taking a certain set of keys (k, in any iterable format) from a node (n1) to another (k2).
        '''

        self.n1=n1
        self.n2=n2
        self.k=k

        self.n1.add_output(self)
        self.n2.add_input(self)
    
    def value_from_upstream(self, k):
        '''
        Obtain a value from an upstream node according to key
        '''

        while not self.n1.summoned:
            self.n1.calculate()
        
        return self.n1.value[k]
    
    def buffer_from_downstream(self):
        '''
        Method to obtain a buffer seed value from a downstream node
        '''

        while not self.n2.summoned:
            self.n2.calculate_buffer()
        
        return self.n2.buffer
    
    def buffer_from_upstream(self):
        '''
        Method to obtain a buffer seed value from an upstream node
        '''

        while not self.n1.summoned:
            self.n1.calculate_buffer(reverse=False)
        
        return {kk:self.n1.buffer[kk] for kk in self.k}
    
    def Jac_from_downstream(self, x, y):
        '''
        Method to pull a Jacobian (dy/dx) from downstream node, given keys for the desired properties
        '''

        if self.n2.Jac is None:
            raise Exception('A Jacobian has been requested from a not yet calculated node in alg. differentiation')

        return self.n2.Jac[y][x]

class node:
    '''
    Class defining a node in an algorithmic differentiation graph.
    '''

    def __init__(self, f, args_to_inds, outs_to_inds, haspassive=False, passive={}):
        '''
        Define a graph node using a function or function set and a dictionary pointing 
        argument keys to argument list indexes
        '''

        self.f=f

        self.passive=passive

        self.args_to_inds=args_to_inds
        self.outs_to_inds=outs_to_inds

        if type(self.args_to_inds)!=dict:
            self.args_to_inds={k:i for i, k in enumerate(self.args_to_inds)}
        
        if type(self.outs_to_inds)!=dict:
            self.outs_to_inds={k:i for i, k in enumerate(self.outs_to_inds)}

        self.summoned=False

        self.haspassive=haspassive

        self.value=None
        self.Jac=None

        self.buffer={}

        self.up_edges=[]
        self.down_edges=[]
    
    def clean_summoning(self):
        '''
        Clean summoning status for node
        '''

        self.summoned=False

    def clean_value_from(self):
        '''
        Clean records of values and Jacobians from the given node to its dependents (recursively)
        '''

        self.summoned=False

        self.value=None
        self.Jac=None

        for de in self.down_edges:
            de.n2.clean_value_from()

    def clean_buffer(self):
        '''
        Clean node buffer
        '''

        self.buffer={}

    def add_input(self, e):
        '''
        Add an input (upstream) edge to list node.up_edges
        '''

        self.up_edges.append(e)
    
    def add_output(self, e):
        '''
        Add an output (downstream) edge to list node.down_edges
        '''

        self.down_edges.append(e)
    
    def calculate(self):
        '''
        Calculate the value according to calls to upstream edges
        '''

        self.summoned=True

        #assembling inputs
        arglist=[None]*len(self.args_to_inds)

        for ue in self.up_edges:
            for k in ue.k:
                arglist[self.args_to_inds[k]]=ue.value_from_upstream(k)

        if self.haspassive:
            self.value, self.Jac=self.f(*tuple(arglist+[self.passive]))
        
        else:
            self.value, self.Jac=self.f(*tuple(arglist))
    
    def set_value(self, v):
        '''
        Set value independently of calculations based on upstream nodes.
        '''
        
        self.summoned=True

        if type(v)==list:
            self.value={k:v[self.outs_to_inds[k]] for k in self.outs_to_inds}
        elif type(v)==dict:
            self.value=v
        else:
            #assuming it's a scalar:
            self.value={self.outs_to_inds.keys()[0]:v}

        for de in self.down_edges:
            de.n2.clean_value_from()

    def calculate_buffer(self, reverse=True, use_diag=True):
        '''
        Calculate a buffer from downstream node seeds (reverse) or upstream node seeds (direct).
        Mode selected through kwarg flag
        '''

        if not self.summoned:
            self.summoned=True
        
            self.buffer={k:None for k in self.outs_to_inds}

            if reverse:
                for de in self.down_edges:
                    seeds=de.buffer_from_downstream()

                    for outk in de.k:
                        for prop, seed in zip(seeds, seeds.values()):
                            self.buffer[outk]=addnone(self.buffer[outk], _matmulnone(de.Jac_from_downstream(outk, prop), seed))

            else:
                for ue in self.up_edges:
                    seeds=ue.buffer_from_upstream()

                    for outk in self.outs_to_inds:
                        for prop, seed in zip(seeds, seeds.values()):
                            self.buffer[outk]=addnone(self.buffer[outk], _matmulnone(self.Jac[outk][prop], seed, transpose=False))

class head(node):
    '''
    A head node for storing input data
    '''

    def __init__(self, outs_to_inds, passive={}):
        '''
        Initialize a head based on a dictionary of input data
        '''

        if type(outs_to_inds)!=dict:
            self.outs_to_inds={k:i for i, k in enumerate(outs_to_inds)}
        else:
            self.outs_to_inds=outs_to_inds
        
        self.passive=passive

        self.down_edges=[]

        self.summoned=False
        self.value=None
    
    def clean_summoning(self):
        super().clean_summoning()
    
    def clean_value_from(self):
        super().clean_value_from()
    
    def clean_buffer(self):
        super().clean_buffer()
    
    def add_output(self, e):
        super().add_output(e)
    
    def calculate(self):
        '''
        I. E. check whether data is available
        '''

        if not self.summoned:
            raise Exception('Attempting to access unset head data')
    
    def set_value(self, v):
        super().set_value(v)
    
    def calculate_buffer(self, reverse=True):
        super().calculate_buffer(reverse=reverse)

class graph:
    '''
    Class containing info about an algorithmic differentiation graph
    '''

    def __init__(self):
        '''
        Instantiate an empty graph
        '''

        self.nodes={}
        self.edges=[]

        self.heads={}
        self.ends={}
    
    def add_node(self, n, name, head=False, end=False):
        '''
        Add a node. head and end kwarg flags define wether it is an input variable, an output variable or 
        a work variable
        '''

        self.nodes[name]=n

        if head:
            self.heads[name]=n
        elif end:
            self.ends[name]=n
    
    def add_edge(self, e):
        '''
        Add an edge
        '''

        self.edges.append(e)
    
    def get_input_data(self, d):
        '''
        Set input data for head nodes based on provided dictionary
        '''

        for k in d:
            self.heads[k].set_value(d[k])
    
    def calculate(self, ends=None):
        '''
        Calculate output data
        '''

        for h in self.heads:
            if not self.heads[h].summoned:
                raise Exception('Head node %s hasn\'t been set, though calculation has been required' % (h,))

        if ends is None:
            ends=self.ends
        else:
            ends={e:self.nodes[e] for e in ends}

        for e in ends:
            ends[e].calculate()

    def clean_summoning(self):
        '''
        Clean summoning status of all graph nodes
        '''

        for n in self.nodes.values():
            n.clean_summoning()
    
    def clean_buffer(self):
        '''
        Clean buffers in all graph nodes
        '''

        for n in self.nodes.values():
            n.clean_summoning()
    
    def clean_values(self):
        '''
        Clean calculation results of all nodes
        '''

        self.clean_summoning()

        for n in self.nodes.values():
            n.value=None
            
            if hasattr(n, 'Jac'):
                n.Jac=None
        
    def set_seed_reverse(self, prop=None, value=None, ends=None):
        '''
        Set seed to identity matrix in node containing the named property and to None in all other end nodes
        '''

        #first of all, clean summoned status and buffers
        self.clean_summoning()
        self.clean_buffer()

        if ends is None:
            ends=self.ends
        else:
            ends={e:self.nodes[e] for e in ends}

        for k in ends:
            if ends[k].value is None or ends[k].Jac is None:
                raise Exception('End node %s hasn\'t had it\'s value calculated, though reverse AD has been required' % (k,))
        
        if value is None:
            for e in ends.values():
                e.buffer={outk:None for outk in e.outs_to_inds}
                
                if prop in e.buffer:
                    e.buffer[prop]=np.ones_like(e.value[prop])

                e.summoned=True
        else:
            for e in ends.values():
                e.buffer={outk:value[outk] if outk in value else None for outk in e.outs_to_inds}

                e.summoned=True
    
    def get_derivs_reverse(self, prop=None, value=None, ends=None, sparse=False):
        '''
        Get a dictionary with the derivatives of a property based on its key
        '''

        self.set_seed_reverse(prop, ends=ends, value=value)

        for n in self.heads.values():
            n.calculate_buffer()

        derivs={}

        for n in self.heads.values():
            derivs.update(n.buffer)
        
        for e, v in zip(derivs, derivs.values()):
            derivs[e]=None if v is None else v.T
        
        return derivs
    
    def set_seed_direct(self, prop, ends=None, value=None):
        '''
        Set seed to identity matrix in node containing the named property and to None in all other end nodes
        '''

        #first of all, clean summoned status and buffers
        self.clean_summoning()
        self.clean_buffer()

        if ends is None:
            ends=self.ends
        else:
            ends={e:self.nodes[e] for e in ends}

        for k in ends:
            if ends[k].value is None or ends[k].Jac is None:
                raise Exception('End node %s hasn\'t had it\'s value calculated, though direct AD has been required' % (k,))
        
        if value is None:
            for h in self.heads.values():
                h.buffer={outk:None for outk in h.outs_to_inds}
                
                if prop in h.buffer:
                    h.buffer[prop]=np.ones_like(h.value[prop])

                h.summoned=True
        else:
            for h in self.heads.values():
                h.buffer={outk:value[outk] if outk in value else None for outk in h.outs_to_inds}

                h.summoned=True
    
    def get_derivs_direct(self, prop, ends=None, value=None):
        '''
        Get a dictionary with the derivatives of an end node
        '''

        self.set_seed_direct(prop, ends=ends, value=value)

        if ends is None:
            ends=self.ends
        else:
            ends={e:self.nodes[e] for e in ends}

        for n in ends.values():
            n.calculate_buffer(reverse=False)

        derivs={}

        for n in ends.values():
            derivs.update(n.buffer)
        
        return derivs
    
    def get_value(self, prop):
        '''
        Extract a given property from values and return the node object in which it has been found. 
        For debugging only
        '''

        for n in self.nodes.values():
            if prop in n.outs_to_inds:
                return n.value[prop], n
        
        raise Exception('Value %s not found' % (prop,))

def mix_vectors(vecs, format='csr'):
    '''
    Recieves vectors in a tuple and mixes them so that components correspondent to 
    a single cell are put next to each other, in the order given by the tuple.
    Also gives the Jacobian of the linear transformation represented by
    the mixing (equal to the LT matrix).

    Example:
    mix_vectors((a, b, c))

    returns:

    {a1, b1, c1, a2, b2, c2...}, J
    '''

    nvars=len(vecs)
    ni=len(vecs[0])

    J=sps.eye(ni*nvars, format=format)

    order=np.hstack([np.arange(i, ni*nvars, nvars, dtype='int') for i in range(nvars)])

    J=J[:, order]

    return J@np.hstack(vecs), J

def reorder_Jacobian(argord, length, format='csr'):
    '''
    Recieves an argument order and returns the Jacobian of a vector reordenation according
    to index order argord. Notice that argord does not necessarily denote the order of the reordered vector.
    '''

    J=sps.eye(length, format=format)

    return J[argord, :]

def diag_cell_indexing(correspondence, nnodes, ncells):
    '''
    Returns indexing for Jacobian transformation from diagonal nodal notation to universal notation, 
    given correspondence
    '''

    return (np.arange(4*ncells, dtype='int'), np.hstack(correspondence.tolist()), ncells, nnodes,)

def diag_cell_Jacobian(J, indexing):
    '''
    Given a diagonal Jacobian in respect to nodal notation (given as a diagonal representing 
    vector), convert it to universal, station-wise notation according to correspondence
    '''

    ncells=indexing[2]
    nnodes=indexing[3]

    return sps.coo_matrix((J, (indexing[0], indexing[1])), shape=(4*ncells, nnodes))

def dcell_dnode_Jacobian(vset, correspondence):
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
    nv=len(vset[0])
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

def LT_node_mix(T):
    '''
    Convert a linear transformation matrix to mixed nodal value format.

    Example: converting A{J11, J12, ..., J33}={Jxx, ..., Jzz} to nodal form would return
    B{J1_11, J1_12, ..., J4_33}={J1_xx, J1_xy, ..., J4_zz}
    '''

    m=np.size(T, axis=0)
    n=np.size(T, axis=1)

    Tnew=np.empty((4*m, 4*n))

    for i in range(4):
        Tnew[i::4, i::4]=T

    return Tnew
