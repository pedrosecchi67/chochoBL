import numpy as np
import scipy.sparse as sps

import pytest

from chochoBL import *

def _arr_compare(a, b, tol=1e-5):
    return np.all(np.abs(a-b)<tol)

def test_function_evaluation():
    '''
    Test for function class creation and derivative evaluation
    '''

    arglist=[np.array([0.0, 1.0]), np.array([1.0, 0.0])]

    f=func(f=lambda x: x@x, derivs=(lambda x: 2*x,), args=[0])
    g=func(f=lambda x: x, derivs=(lambda x: np.eye(len(x)),), args=[1])
    h=func(f=lambda x, y: x*y, derivs=(lambda x, y: np.diag(y), lambda x, y: np.diag(x)), args=[0, 1])

    assert f(arglist)==1.0, "f evaluation failed"
    assert np.all(g(arglist)==np.array([1.0, 0.0])), "g evaluation failed"

    assert np.all(f.Jacobian(arglist)==np.array([0.0, 2.0])), "f Jacobian evaluation failed"
    assert np.all(g.Jacobian(arglist)==np.eye(2)), "g Jacobian evaluation failed"

    fset=funcset(fs=[f, g, h], arglens=[2, 2], outlens=[1, 2, 2])

    fval=fset(arglist)

    assert np.all(fval==np.array([1.0, 1.0, 0.0, 0.0, 0.0])), "Function set evaluation failed"
    assert np.all(
        fset.Jacobian(arglist)==np.array(
            [
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        )
    ), "Function set Jacobian evaluation failed"

    seplist=fset.out_unpack(fval)

    assert (
        np.all(seplist[0]==np.array([1.0])) and \
            np.all(seplist[1]==np.array([1.0, 0.0])) and \
                np.all(seplist[2]==np.array([0.0, 0.0]))
    ), "Function set output separation failed"

def test_vector_mixing():
    a=np.arange(0, 3, 1, dtype='int')
    b=np.arange(3, 6, 1, dtype='int')

    v, J=mix_vectors((a, b))

    assert np.all(v==np.array([0, 3, 1, 4, 2, 5])), "Vector mixing failed"

def test_vector_reordering_Jacobian():
    a=np.array([0, 1, 2])
    b=np.array([1, 0, 2])

    J=reorder_Jacobian(b, 3)

    assert np.all(J@a==np.array([1, 0, 2])), "Vector reordering Jacobian failed"

def test_correspondence_Jacobian():
    a=np.arange(0.0, 12.0, 2.0)
    b=np.arange(1.0, 13.0, 2.0)

    correspondence=np.array(
        [
            [0, 1, 4, 3],
            [1, 2, 5, 4]
        ], dtype='int'
    )

    J=dcell_dnode_Jacobian((a, b), correspondence)

    ideal_cell_mix=np.array(
        [0.0, 2.0, 8.0, 6.0, 1.0, 3.0, 9.0, 7.0, 2.0, 4.0, 10.0, 8.0, 3.0, 5.0, 11.0, 9.0]
    )

    assert np.all(J@np.hstack((a, b))==ideal_cell_mix), "Node to cell Jacobian test failed"

def test_LT_node_mix():
    T=np.array(
        [
            [1.0, 2.0, 0.0],
            [1.0, 0.0, -9.0],
            [5.0, -1.0, 0.0]
        ]
    )

    a=np.array([0.0, 1.0, -1.0])
    b=np.array([1.0, 0.0, -1.0])
    c=np.array([4.0, 1.0, -4.0])
    d=np.array([1.0, 0.0, 0.0])

    Ta=T@a
    Tb=T@b
    Tc=T@c
    Td=T@d

    Tnew=LT_node_mix(T)
    mixvars, _=mix_vectors((a, b, c, d))
    mixres, _=mix_vectors((Ta, Tb, Tc, Td))

    assert np.all(Tnew@mixvars==mixres), "LT node mixing test failed"

def test_chain():
    f=func(f=lambda x: x**2, derivs=(lambda x: 2*np.diag(x),), args=[0])
    g=func(f=lambda y: np.sqrt(y), derivs=(lambda y: 1.0/(2*np.sqrt(y)),), args=[1])

    def t(x, y):
        v=np.hstack((x, y))

        return np.array([v@v])
    
    def dt_dx(x, y):
        return np.hstack((np.ones_like(x), np.zeros_like(y)))
    
    def dt_dy(x, y):
        return np.hstack((np.zeros_like(x), np.ones_like(y)))
    
    transf=func(f=t, derivs=(dt_dx, dt_dy,), args=[0, 1])

    fs=funcset(fs=[f, g], arglens=[2, 2], outlens=[2, 2])

    h=func(f=lambda x: x**2, derivs=(lambda x: 2*x,), args=[0])

    ch=chain(f=h, transfer=transf)
    ch2=chain(f=ch, transfer=fs)

    arglist=[np.array([1.0, 2.0]), np.array([3.0, 4.0])]

    assert ch2(arglist)==576.0, "Chain rule evaluation failed"

def test_graph_calculate():
    passive={'k':2, 'l':3}

    f=head(outs_to_inds={'x':0, 'y':1, 'z':2}, passive=passive)

    def af(y, z, passive={}):
        return y**2*z/passive['k']
    def daf_dy(y, z, passive={}):
        return 2*sps.diags(y*z)/passive['k']
    def daf_dz(y, z, passive={}):
        return sps.diags(y**2)/passive['k']
    
    a=func(f=af, derivs=(daf_dy, daf_dz,), haspassive=True, args=[0, 1], sparse=True)
    
    def bf(y, z, passive={}):
        return y*z**2/passive['l']
    def dbf_dy(y, z, passive={}):
        return sps.diags(z**2)/passive['l']
    def dbf_dz(y, z, passive={}):
        return 2*sps.diags(y*z)/passive['l']
    
    b=func(f=bf, derivs=(dbf_dy, dbf_dz,), haspassive=True, args=[0, 1], sparse=True)

    gfset=funcset(fs=[a, b], arglens=[1, 1], outlens=[1, 1], sparse=True)

    g=node(f=gfset, args_to_inds=['y', 'z'], outs_to_inds=['a', 'b'], passive=passive)

    def hf(a):
        return a**3
    def dhf_da(a):
        return 3*sps.diags(a**2)
    
    hfunc=func(f=hf, derivs=(dhf_da,), haspassive=False, args=[0], sparse=True)
    
    h=node(f=hfunc, args_to_inds={'a'}, outs_to_inds={'h'}, passive=passive)

    def kf(b):
        return b**2
    def dkf_db(b):
        return 2*sps.diags(b)
    
    kfunc=func(f=kf, derivs=(dkf_db,), haspassive=False, args=[0], sparse=True)

    k=node(f=kfunc, args_to_inds={'b'}, outs_to_inds={'k'}, passive=passive)

    gr=graph()

    gr.add_node(f, 'f', head=True)
    gr.add_node(g, 'g')
    gr.add_node(h, 'h', end=True)
    gr.add_node(k, 'k', end=True)

    e1=edge(f, g, k={'y', 'z'})

    e2=edge(g, h, k={'a'})

    e3=edge(g, k, k={'b'})

    gr.add_edge(e1)
    gr.add_edge(e2)
    gr.add_edge(e3)

    f.set_value({'x':np.array([1.0]), 'y':np.array([2.0]), 'z':np.array([3.0])})

    gr.calculate()

    assert _arr_compare(h.value['h'], 216.0) and _arr_compare(k.value['k'], 36.0), "Graph function evaluation failed"
    assert _arr_compare(g.Jac['a']['y'], np.array([[6.0]])) and \
        _arr_compare(g.Jac['a']['z'], np.array([[2.0]])) and \
            _arr_compare(g.Jac['b']['y'], np.array([[3.0]])) and \
                _arr_compare(g.Jac['b']['z'], np.array([[4.0]])), "Node Jacobian evaluation failed"

def test_graph_derivate():
    x=head(outs_to_inds=['x'])
    
    y=head(outs_to_inds=['y'])

    def w1f(x, y):
        return x*y
    def dw1f_dx(x, y):
        return np.diag(y)
    def dw1f_dy(x, y):
        return np.diag(x)
    
    w1func=func(f=w1f, args=[0, 1], derivs=(dw1f_dx, dw1f_dy,))

    w1=node(f=w1func, args_to_inds=['x', 'y'], outs_to_inds=['w1'])

    def w2f(x, y):
        return x+y
    def dw2f_dx(x, y):
        return np.array([[1.0]])
    def dw2f_dy(x, y):
        return np.array([[1.0]])

    w2func=func(f=w2f, args=[0, 1], derivs=(dw2f_dx, dw2f_dy,))

    w2=node(f=w2func, args_to_inds=['x', 'y'], outs_to_inds=['w2'])

    def gf(w1, w2):
        return w1+w2
    def dgf_dw1(w1, w2):
        return np.array([[1.0]])
    def dgf_dw2(w1, w2):
        return np.array([[1.0]])
    
    gfunc=func(f=gf, derivs=(dgf_dw1, dgf_dw2,), args=[0, 1])

    g=node(f=gfunc, outs_to_inds=['g'], args_to_inds=['w1', 'w2'])

    def ff(w1, w2):
        return w1+np.exp(w2)
    def dff_dw1(w1, w2):
        return np.array([[1.0]])
    def dff_dw2(w1, w2):
        return np.diag(np.exp(w2))
    
    ffunc=func(f=ff, derivs=(dff_dw1, dff_dw2,), args=[0, 1])

    f=node(f=ffunc, outs_to_inds=['f'], args_to_inds=['w1', 'w2'])

    gr=graph()

    x_w1=edge(x, w1, {'x'})
    x_w2=edge(x, w2, {'x'})
    
    y_w1=edge(y, w1, {'y'})
    y_w2=edge(y, w2, {'y'})

    w1_f=edge(w1, f, {'w1'})
    w2_f=edge(w2, f, {'w2'})

    w1_g=edge(w1, g, {'w1'})
    w2_g=edge(w2, g, {'w2'})

    gr.add_node(x, 'x', head=True)
    gr.add_node(y, 'y', head=True)
    gr.add_node(w1, 'w1')
    gr.add_node(w2, 'w2')
    gr.add_node(f, 'f', end=True)
    gr.add_node(g, 'g', end=True)

    x.set_value({'x':np.array([1.0])})
    y.set_value({'y':np.array([2.0])})

    gr.calculate()

    derivs_f=gr.get_derivs('f')
    derivs_g=gr.get_derivs('g')

    assert _arr_compare(derivs_f['x'], np.exp(3)+2) and \
        _arr_compare(derivs_f['y'], np.exp(3)+1) and \
            _arr_compare(derivs_g['x'], 3) and \
                _arr_compare(derivs_g['y'], 2)

def test_graph_funcset():
    xv=np.array([1.0, 2.0])
    yv=np.array([-2.0, -3.0])

    q=np.array([5.0, 4.0])

    P=np.array(
        [
            [1.0, -1.0],
            [3.0, 2.0]
        ]
    )

    def ff(x, y):
        return x@P@y
    def dff_dx(x, y):
        return (P@y).reshape((1, 2))
    def dff_dy(x, y):
        return (x@P).reshape((1, 2))
    
    def gf(y):
        return q@y
    def dgf_dy(y):
        return q.reshape((1, 2))
    
    ffunc=func(f=ff, derivs=(dff_dx, dff_dy,), args=[0, 1])
    gfunc=func(f=gf, args=[1], derivs=(dgf_dy,))

    fset=funcset(fs=[ffunc, gfunc], arglens=[2, 2], outlens=[1, 1])

    gr=graph()

    n1=head(outs_to_inds=['x', 'y'])
    n2=node(f=fset, args_to_inds=['x', 'y'], outs_to_inds=['f', 'g'])

    e1=edge(n1, n2, {'x', 'y'})

    gr.add_node(n1, 'n1', head=True)
    gr.add_node(n2, 'n2', end=True)
    
    gr.add_edge(e1)

    n1.set_value({'x':xv, 'y':yv})

    gr.calculate()

    derivs_f=gr.get_derivs('f')
    derivs_g=gr.get_derivs('g')

    assert _arr_compare(derivs_f['x'], P@yv) and _arr_compare(derivs_f['y'], xv@P) and _arr_compare(derivs_g['y'], q), "Function set derivation failed"

test_graph_funcset()