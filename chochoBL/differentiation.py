import numpy as np
import scipy.sparse as sps

'''
Module with functions and classes to assist in Jacobian handling
and automatic differentiation
'''

class func:
    '''
    Class defining as function application, with argument list
    '''

    def __init__(self, f, derivs, args):
        '''
        Class containing info about a given function, recieving its argument indexes
        '''

        self.f=f
        self.derivs=derivs
        self.args=args #list of argument indexes
    
    def __call__(self, arglist):
        '''
        Return the evaluation of a function according to arguments in arglist with their indexes listed
        in self.args
        '''

        return self.f(*tuple(arglist[i] for i in self.args))

    def Jacobian(self, arglist):
        '''
        Return the evaluation of a function's Jacobian according to arguments in arglist with their indexes listed
        in self.args
        '''

        return np.hstack([d(*tuple([arglist[i] for i in self.args])) for d in self.derivs])

class funcset:
    '''
    Class defining a set of functions
    '''

    def __init__(self, fs=[], arglens=[], outlens=[]):
        '''
        Define a set of functions based on a list of its function classes and a list of it's argument array lengths
        and output array lengths
        '''

        self.fs=fs

        self.argn=sum(arglens)
        self.outn=sum(outlens)
        
        lastinds=[]
        for i, l in enumerate(arglens):
            if i==0:
                lastinds.append(l)
            else:
                lastinds.append(l+lastinds[-1])
        
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

    def __call__(self, arglist):
        '''
        Evaluate a function and return its output as a vector
        '''

        return np.hstack(
            [
                f(arglist) for f in self.fs
            ]
        )
    
    def Jacobian(self, arglist, mtype='dense'):
        '''
        Returns the Jacobian as a function. Kwarg mtype identifies type of matrix to be returned 
        (csr_matrix, csc_matrix, lil_matrix or dense, passed as string).
        '''

        shape=(self.outn, self.argn)

        if mtype=='csr_matrix':
            J=sps.csr_matrix(shape)
        elif mtype=='csc_matrix':
            J=sps.csc_matrix(shape)
        elif mtype=='lil_matrix':
            J=sps.lil_matrix(shape)
        elif mtype=='dense':
            J=np.zeros(shape)
        else:
            raise Exception('Matrix type for function set Jacobian not identified')
        
        for f, (argi, outi) in zip(self.fs, zip(self.arginds, self.outinds)):
            J[outi[0]:outi[1], argi]=f.Jacobian(arglist)
        
        return J

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

def _tensor_convert(T, Mtosys):
    return Mtosys.T@T@Mtosys

def tensor_conversion_Jacobian(Mtosys):
    '''
    Function to return a coordinate system conversion Jacobian for a 3x3 tensor (inputed as a length 9 vector)
    '''

    v=np.empty(9)
    J=np.zeros((9, 9))

    for i in range(9):
        v[:]=0.0
        v[i]=1.0

        J[:, i]=_tensor_convert(v.reshape((3, 3)), Mtosys).reshape(9)
    
    return J

def tensor_convert_to_bidim(Mtosys):
    '''
    Function to return a coordinate system conversion Jacobian from a 3x3 tensor to a 2x2 bidimensional considering
    only axes x and z
    '''

    C=tensor_conversion_Jacobian(Mtosys)

    return C[np.array([0, 2, 6, 8], dtype='int'), :]

def LT_node_mix(T):
    '''
    Convert a linear transformation matrix to mixed nodal value format.

    Example: converting A{J11, J12, ..., J33}={Jxx, ..., Jzz} to nodal form would return
    B{J1_11, J2_11, ..., J4_33}={J1_xx, J2_xx, ..., J4_zz}
    '''

    m=np.size(T, axis=0)
    n=np.size(T, axis=1)

    Tnew=np.empty((4*m, 4*n))

    for i in range(4):
        Tnew[i::4, i::4]=T

    return Tnew
