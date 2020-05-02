import os
import sys
import py_compile
ordir=os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
#module imports
#these lines ensure pycaching prior to module importation
py_compile.compile('abaqus.py')
py_compile.compile('station.py')
from abaqus import *
from closure import *
from station import *
os.chdir(ordir)
del ordir