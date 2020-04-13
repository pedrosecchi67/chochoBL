import os
import sys
ordir=os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
#module imports
import HS_solver
os.chdir(ordir)
del ordir