# chochoBL

ChoCho BL is a Python based three-dimensional boundary layer solver to be used in conjunction with panel methods.
Its Python frontend is customized so as to ease interaction with a varied range of panel method codes and discretization techniques, while a powerful finite-difference FORTRAN-programmed engine provides fast computations to match the speed requirements of viscid-inviscid coupling applications. Easy access to Python source code should also guarantee an easily customizable code, adaptable to several different theoretical models for turbulence prediction and modelling and discretization settings.

# Introduction

This product is being developed so as to meet the following system requirements, numbered by their goal, as:

(1) provide quick computation of finite-difference solutions for RANS in any solid's boundary layer;
(2) provide quick adaptability of employed theoretical models for the user's own precision and performance requirements, as well as to ease application for different academic and engineering purposes;
(3) provide easy input and output functionalities.

* Have directly accessible mesh definition functions for variable precision and computation time; (1, 2)
* Accept custom definitions for turbulence criteria and application in viscous drag (custom eddy viscosity distribution and forced transition x positions); (2)
* Being able to perform boundary layer solutions with either a streamwise solution of x-momentum equation or a full 3D boundary layer equations solution, depending on project requirements for analysis precision; (2)
* Define key aerodynamic variables (BL thicknesses, local shear stresses and velocity profiles) as extractable python variables/class atributes. (3)

It has been planned to work as a viscosity calculation module for LovelacePM (https://github.com/pedrosecchi67/LovelacePM) and designed so as to be adaptable to many different fashions of 3D panel codes and aerodynamic frameworks with ease.

### Test cases

ChoCho Boundary Layer solver will have the following test cases as readiness criteria for its beta version (v1.X.Y):

* Airfoils NACA-0012 and SD-7062 airfoils as output from Mark Drela's Xfoil code, evaluated by differences in local pressure coefficient distribution and local boundary layer thickness distribution;
* Onera M6 wing as proposed in https://turbmodels.larc.nasa.gov/onerawingnumerics_val_sa.html, evaluated in precision of coefficients when used in conjunction with LovelacePM.

### Code conventions

* Python's suggested naming conventions are adopted. Initials may also be referred to in capital letters;
* All FORTRAN backend subroutines should be defined in toolkit.so, obeying the FORTRAN 90 standard.