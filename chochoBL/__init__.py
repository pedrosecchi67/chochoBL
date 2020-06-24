import os
import sys
import py_compile
ordir=os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
#module imports
#these lines ensure pycaching prior to module importation
py_compile.compile('CG.py')
py_compile.compile('closure.py')
py_compile.compile('transition.py')
py_compile.compile('differentiation.py')
py_compile.compile('three_equation.py')
py_compile.compile('mesh.py')
py_compile.compile('garlekin.py')
py_compile.compile('mapping.py')
py_compile.compile('adjoint.py')
from CG import *
from closure import *
from transition import *
from differentiation import *
from three_equation import *
from mesh import *
from garlekin import *
from mapping import *
from adjoint import *
os.chdir(ordir)
del ordir