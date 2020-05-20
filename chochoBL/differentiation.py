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
        (csr_matrix, csc_matrix, lil_matrix or dense, passed as string)
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