import numpy as np
import scipy.sparse as sps

'''
Module with functions and classes to assist in Jacobian handling
and automatic differentiation
'''

def _addnone(a, b):
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

def _matmulnone(A, B):
    if A is None or B is None:
        return None
    else:
        return A@B

class func:
    '''
    Class defining as function application, with argument list
    '''

    def __init__(self, f, derivs, args, haspassive=False, sparse=False):
        '''
        Class containing info about a given function, recieving its argument indexes.
        Passive should be a dictionary containing variables to be called without
        being regarded as differentiation variables. haspassive should be a boolean indicating if the function should recieve a passive.
        Treats matrixes as sparse (using scipy.sps) if sparse is set to True
        '''

        self.f=f
        self.derivs=derivs
        self.args=args #list of argument indexes
        self.haspassive=haspassive
        self.sparse=sparse
    
    def __call__(self, arglist, passive={}):
        '''
        Return the evaluation of a function according to arguments in arglist with their indexes listed
        in self.args.
        Passive should be a dictionary containing variables to be called without
        being regarded as differentiation variables
        '''

        if self.haspassive:
            return self.f(*(tuple(arglist[i] for i in self.args)+(passive,)))
        else:
            return self.f(*tuple(arglist[i] for i in self.args))

    def Jacobian(self, arglist, mtype='dense', passive={}):
        '''
        Return the evaluation of a function's Jacobian according to arguments in arglist with their indexes listed
        in self.args.
        Passive should be a dictionary containing variables to be called without
        being regarded as differentiation variables
        '''

        if self.sparse:
            if mtype=='dense':
                raise Exception('Function object set to sparse mode, but Jacobian required as dense')
            
            if self.haspassive:
                return sps.hstack([d(*(tuple(arglist[i] for i in self.args)+(passive,))) for d in self.derivs], format=mtype)
            else:
                return sps.hstack([d(*tuple(arglist[i] for i in self.args)) for d in self.derivs], format=mtype)
        
        else:
            if mtype!='dense':
                raise Exception('Function object set to dense mode, but Jacobian required in format %s' % (mtype,))

            if self.haspassive:
                return np.hstack([d(*(tuple(arglist[i] for i in self.args)+(passive,))) for d in self.derivs])
            else:
                return np.hstack([d(*tuple(arglist[i] for i in self.args)) for d in self.derivs])

class funcset:
    '''
    Class defining a set of functions
    '''

    def __init__(self, fs=[], arglens=[], outlens=[], sparse=False):
        '''
        Define a set of functions based on a list of its function classes and a list of it's argument array lengths
        and output array lengths. Treats matrixes according to scipy.sparse if sparse kwarg flag is set to True
        '''

        self.fs=fs

        self.sparse=sparse

        self.argn=sum(arglens)
        self.outn=sum(outlens)
        
        lastinds=[]
        for i, l in enumerate(arglens):
            if i==0:
                lastinds.append(l)
            else:
                lastinds.append(l+lastinds[-1])

        self.arglims=[[0 if i==0 else lastinds[i-1], lastinds[i]] for i in range(len(arglens))]
        
        arginds=[np.arange(0 if i==0 else lastinds[i-1], lastinds[i], 1, dtype='int') for i in range(len(arglens))]

        self.arginds=[]
        for f in self.fs:
            inds=np.array([], dtype='int')
            for i in f.args:
                inds=np.hstack((inds, arginds[i]))

            self.arginds.append(inds)

        lastinds=[]
        for i, l in enumerate(outlens):
            if i==0:
                lastinds.append(l)
            else:
                lastinds.append(l+lastinds[-1])
        
        self.outinds=[[0 if i==0 else lastinds[i-1], lastinds[i]] for i in range(len(fs))]

    def __call__(self, arglist, passive={}):
        '''
        Evaluate a function and return its output as a vector.
        Passive should be a dictionary containing variables to be called without
        being regarded as differentiation variables
        '''

        return np.hstack(
            [
                f(arglist, passive=passive) for f in self.fs
            ]
        )
    
    def Jacobian(self, arglist, mtype='dense', passive={}):
        '''
        Returns the Jacobian as a function. Kwarg mtype identifies type of matrix to be returned 
        (csr, csc, lil or dense, passed as string).
        Passive should be a dictionary containing variables to be called without
        being regarded as differentiation variables
        '''

        shape=(self.outn, self.argn)

        if mtype=='csr':
            J=sps.csr_matrix(shape)
            conv=lambda x: sps.csr_matrix(x)
        elif mtype=='csc':
            J=sps.csc_matrix(shape)
            conv=lambda x: sps.csc_matrix(x)
        elif mtype=='lil':
            J=sps.lil_matrix(shape)
            conv=lambda x: sps.lil_matrix(x)
        elif mtype=='dense':
            J=np.zeros(shape)
            conv=lambda x: x.todense()
        else:
            raise Exception('Matrix type for function set Jacobian not identified')

        if self.sparse and mtype=='dense':
            raise Exception('Function set specified as sparse, but Jacobian requested as dense')
        
        for f, (argi, outi) in zip(self.fs, zip(self.arginds, self.outinds)):
            jac=f.Jacobian(arglist, mtype=mtype, passive=passive)

            if self.sparse:
                J[outi[0]:outi[1], argi]=jac if sps.issparse(jac) else conv(jac)
            
            else:
                J[outi[0]:outi[1], argi]=jac.todense() if sps.issparse(jac) else jac
        
        return J

    def out_unpack(self, f):
        '''
        Recieves an output in the form of a stacked vector and decomposes it into the individual functions
        that formed it.
        '''

        return [f[outi[0]:outi[1]] for outi in self.outinds]
    
    def in_unpack(self, x):
        '''
        Recieves an input in the form of a stacked vector and decomposes it into the individual
        arguments that formed it.
        '''

        arguments=[x[lim[0]:lim[1]] for lim in self.arglims]

        return [float(a) if np.size(a)==1 else a for a in arguments]
    
    def J_unpack(self, J):
        '''
        Recieves as input a Jacobian in matricial form and returns a list of lists with each block component,
        according to a function-argument correspondence.
        '''

        return [
            [J[outi[0]:outi[1], argi[0]:argi[1]] for argi in self.arglims] for outi in self.outinds
        ]

class chain:
    '''
    Class to contain information about a change in variables passed to a function/function set.
    '''

    def __init__(self, f, transfer):
        '''
        Instantiate a chain class object. See help(chain) for more info.

        * f:
        function object to be wrapped in this class\'s instance
        * transfer: 
        function such that f(transfer(args, [passive]), [passive]) is to be returned by __call__

        When a Jacobian is to be computed, chain.Jacobian() method should return f.Jacobian()@transfer.Jacobian()
        '''

        self.f=f
        self.transfer=transfer
    
    def __call__(self, arglist, passive={}):
        '''
        Return f(transfer(args, [passive]), [passive])
        '''

        newargs=self.transfer(arglist, passive)

        if type(self.transfer)==funcset:
            newargs=self.transfer.out_unpack(newargs)
        
        return self.f(newargs, passive=passive)
    
    def Jacobian(self, arglist, passive={}):
        '''
        Return f.Jacobian()@transfer.Jacobian()
        '''

        newargs=self.transfer.out_unpack(self.transfer(arglist, passive))

        return self.f.Jacobian(newargs, passive=passive)@self.transfer.Jacobian(arglist, passive=passive)
    
    def out_unpack(self, f):
        '''
        Unpack results in the form of stacked vectors
        '''

        return self.f.out_unpack(f)
    
    def in_unpack(self, x):
        '''
        Unpack arguments in the form of stacked vectors
        '''

        return self.transfer.in_unpack(x)

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

    def __init__(self, f, args_to_inds, outs_to_inds, passive={}):
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

        if type(self.f)!=func or self.f.haspassive:
            mtype='lil' if self.f.sparse else 'dense'

            output=self.f(arglist, passive=self.passive)
            jac=self.f.Jacobian(arglist, mtype=mtype, passive=self.passive)
        else:
            mtype='lil' if self.f.sparse else 'dense'
            
            output=self.f(arglist)
            jac=self.f.Jacobian(arglist, mtype=mtype)
        
        if type(self.f)==func:
            output=[output]
        else:
            output=self.f.out_unpack(output)
        
        #convert output to dictionary form
        self.value={k:output[self.outs_to_inds[k]] for k in self.outs_to_inds}

        #convert Jacobian to dictionary form
        #calculate argument and output limits
        if type(self.f)==funcset:
            arglims=self.f.arglims
            outlims=self.f.outinds
        else:
            arglims=[]
            outlims=[]

            for i, arg in enumerate(arglist):
                if i==0:
                    arglims.append([0, len(arg)])
                else:
                    arglims.append([arglims[-1][1], arglims[-1][1]+len(arg)])
            
            for i, out in enumerate(output):
                if i==0:
                    outlims.append([0, len(out)])
                else:
                    outlims.append([outlims[-1][1], outlims[-1][1]+len(out)])
            
        self.Jac={}

        for outn in self.outs_to_inds:
            self.Jac[outn]={}
            outl=outlims[self.outs_to_inds[outn]]

            for argn in self.args_to_inds:
                argl=arglims[self.args_to_inds[argn]]

                self.Jac[outn][argn]=jac[outl[0]:outl[1], argl[0]:argl[1]]
    
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
    
    def calculate_buffer(self):
        '''
        Calculate a buffer from downstream node seeds
        '''

        self.summoned=True

        self.buffer={k:None for k in self.outs_to_inds}

        for de in self.down_edges:
            seeds=de.buffer_from_downstream()

            for outk in de.k:
                for prop, seed in zip(seeds, seeds.values()):
                    self.buffer[outk]=_addnone(self.buffer[outk], _matmulnone(de.Jac_from_downstream(outk, prop).T, seed))

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
    
    def calculate_buffer(self):
        super().calculate_buffer()

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
    
    def calculate(self):
        '''
        Calculate output data
        '''

        for h in self.heads:
            if not self.heads[h].summoned:
                raise Exception('Head node %s hasn\'t been set, though calculation has been required' % (h,))

        for e in self.ends:
            self.ends[e].calculate()

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
        
    def set_seed(self, prop, sparse=False):
        '''
        Set seed to identity matrix in node containing the named property and to None in all other end nodes
        '''

        #first of all, clean summoned status and buffers
        self.clean_summoning()
        self.clean_buffer()

        for k in self.ends:
            if self.ends[k].value is None or self.ends[k].Jac is None:
                raise Exception('End node %s hasn\'t had it\'s value calculated, though reverse AD has been required' % (k,))
        
        for e in self.ends.values():
            e.buffer={outk:None for outk in e.outs_to_inds}
            
            if prop in e.buffer:
                e.buffer[prop]=sps.eye(len(e.value[prop]), format='lil') if sparse else np.eye(len(e.value[prop]))

            e.summoned=True
    
    def get_derivs(self, prop, sparse=False):
        '''
        Get a dictionary with the derivatives of a property based on its key
        '''

        self.set_seed(prop, sparse=sparse)

        for n in self.heads.values():
            n.calculate_buffer()

        derivs={}

        for n in self.heads.values():
            derivs.update(n.buffer)
        
        for e, v in zip(derivs, derivs.values()):
            derivs[e]=None if v is None else v.T
        
        return derivs

def identity(args, format='lil', haspassive=False):
    '''
    Define an identity function object according to matrix format for Jacobian
    '''
    if format!='dense':
        return func(f=lambda x: x, derivs=(lambda x: sps.eye(len(x), format=format),), args=args, haspassive=haspassive, sparse=True)
    
    else:
        return func(f=lambda x: x, derivs=(lambda x: np.eye(len(x)),), args=args, haspassive=haspassive, sparse=False)

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

def dcell_dnode_Jacobian(vset, correspondence):
    '''
    Given a set of vectors corresponding to a set of nodes and an array of correspondences (shape (nnodes, 4))
    between cell and node indexes, returns 4 vectors (corresponding to the set of indexes 1, 2, 3, and 4 in the
    cells) and the Jacobian of their combination.

    Argument correspondence corresponds to a shape (ncells, 4) array with cell.indset indexes

    The Jacobian is returned with respect to the stacked vectors, not to any mixing.

    Example:
    given (a, b, c), returns:
    {a11, a12, a13, a14, b11, b12, b13, b14, ..., cnm}, J
    '''

    nprop=len(vset)
    nv=len(vset[0])
    ncells=np.size(correspondence, axis=0)

    J=sps.lil_matrix((ncells*4*nprop, nv*nprop))

    for i in range(4):
        for j in range(nprop):
            for k in range(ncells):
                J[k*4*nprop+4*j+i, correspondence[k, i]+j*nv]=1.0
    
    return J

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
