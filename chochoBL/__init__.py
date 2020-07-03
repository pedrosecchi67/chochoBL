import os
import sys

ordir=os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from mesh import *
from adjoint import *
from residual import *
from three_equation import *
from three_equation_b import *
from three_equation_d import *

os.chdir(ordir)
del ordir