import numpy as np
import scipy.sparse as sps

import pytest

from chochoBL import *

def _arr_compare(a, b, tol=1e-5):
    return np.all(np.abs(a-b)<tol)

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

def test_graph_calculate():
    passive={'k':2, 'l':3}

    f=head(outs_to_inds={'x':0, 'y':1, 'z':2}, passive=passive)

    def af(y, z, passive={}):
        return y**2*z/passive['k']
    def daf_dy(y, z, passive={}):
        return 2*sps.diags(y*z)/passive['k']
    def daf_dz(y, z, passive={}):
        return sps.diags(y**2)/passive['k']
    
    def bf(y, z, passive={}):
        return y*z**2/passive['l']
    def dbf_dy(y, z, passive={}):
        return sps.diags(z**2)/passive['l']
    def dbf_dz(y, z, passive={}):
        return 2*sps.diags(y*z)/passive['l']
    
    def gcalc(y, z, passive):
        values={}
        Jac={}

        values['a']=af(y, z, passive)
        values['b']=bf(y, z, passive)

        Jac['a']={'y':daf_dy(y, z, passive), 'z':daf_dz(y, z, passive)}
        Jac['b']={'y':dbf_dy(y, z, passive), 'z':dbf_dz(y, z, passive)}

        return values, Jac

    g=node(f=gcalc, args_to_inds=['y', 'z'], outs_to_inds=['a', 'b'], haspassive=True, passive=passive)

    def hf(a):
        return a**3
    def dhf_da(a):
        return 3*sps.diags(a**2)

    def hcalc(a):
        value={}
        Jac={}

        value['h']=hf(a)
        Jac['h']={'a':dhf_da(a)}

        return value, Jac
    
    h=node(f=hcalc, args_to_inds={'a'}, outs_to_inds={'h'}, haspassive=False, passive=passive)

    def kf(b):
        return b**2
    def dkf_db(b):
        return 2*sps.diags(b)
    
    def kcalc(b):
        value={}
        Jac={}

        value['k']=kf(b)
        Jac['k']={'b':dkf_db(b)}

        return value, Jac

    k=node(f=kcalc, args_to_inds={'b'}, outs_to_inds={'k'}, haspassive=False, passive=passive)

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
    
    def w1func(x, y):
        value={'w1':w1f(x, y)}
        Jac={'w1':{'x':dw1f_dx(x, y), 'y':dw1f_dy(x, y)}}

        return value, Jac

    w1=node(f=w1func, args_to_inds=['x', 'y'], outs_to_inds=['w1'])

    def w2f(x, y):
        return x+y
    def dw2f_dx(x, y):
        return np.array([[1.0]])
    def dw2f_dy(x, y):
        return np.array([[1.0]])

    def w2func(x, y):
        value={'w2':w2f(x, y)}
        Jac={'w2':{'x':dw2f_dx(x, y), 'y':dw2f_dy(x, y)}}

        return value, Jac

    w2=node(f=w2func, args_to_inds=['x', 'y'], outs_to_inds=['w2'])

    def gf(w1, w2):
        return w1+w2
    def dgf_dw1(w1, w2):
        return np.array([[1.0]])
    def dgf_dw2(w1, w2):
        return np.array([[1.0]])
    
    def gfunc(w1, w2):
        value={'g':gf(w1, w2)}
        Jac={'g':{'w1':dgf_dw1(w1, w2), 'w2':dgf_dw2(w1, w2)}}

        return value, Jac

    g=node(f=gfunc, outs_to_inds=['g'], args_to_inds=['w1', 'w2'])

    def ff(w1, w2):
        return w1+np.exp(w2)
    def dff_dw1(w1, w2):
        return np.array([[1.0]])
    def dff_dw2(w1, w2):
        return np.diag(np.exp(w2))
    
    def ffunc(w1, w2):
        value={'f':ff(w1, w2)}
        Jac={'f':{'w1':dff_dw1(w1, w2), 'w2':dff_dw2(w1, w2)}}

        return value, Jac

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
