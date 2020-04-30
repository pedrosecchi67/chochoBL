import os
import sys
import py_compile
ordir=os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
#module imports
py_compile.compile('abaqus.py')
py_compile.compile('station.py')
import abaqus
import closure
import station
os.chdir(ordir)
del ordir