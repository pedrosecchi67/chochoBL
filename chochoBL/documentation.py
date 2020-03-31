import network

network.__doc__='''
Module encompassing information about a network and its point grid, upon which to exercise BL calculations

=======
classes
=======

network: class encompassing information about a grid point and local velocity values, with respective calculation methods
'''

network.network.__doc__='''
Class encompassing information about a grid point and local velocity values, with respective calculation methods

NOTICE: in order to discretize the grid geometry and ease the application of finite difference differentiations,
auxiliary variables lambda_x, lambda_y and lambda_t (for thickness) -- or lx, ly and lt -- are employed. Those variables
vary uniformly accross their respective dimension along the grid and derivatives dxdlx, dydlx, ... are employable to find surface
gradients and normal derivatives through using finite difference derivatives and chain rule.

=========
variables
=========

rho: fluid density for BL simulations
mu: fluid viscosity for BL simulations
idisc, jdisc, ndisc: size of the grid matrixes in streamwise, crosswash and normal directions, respectively
origin: array (shape (idisc, jdisc, 3)) identifying an origin for each of the grid nodes
qes: external, inviscid solution velocities (shape (idisc, jdisc, 3)) in universal coordinate system
thicks: normal positions of each control point, offset from respective cell origin in network.origin
dxdlx, dydlx, ... dzdly: derivatives of surface node coordinates in respect to auxiliary coordinates l(amda_)x and lambda_y. Used to compute
surface gradients
Mtosys, Mtouni: arrays (shape (idisc, jdisc, 3, 3)) made to store local coordinate system matrixes
pressgrad: local pressure gradient (array, shape (idisc, jdisc)). Uses uniform boundary layer pressure approximation in normal axis
dlx, dly, dlt: unitary variations in auxiliary coordinates. Used for finite difference derivatives
dydly: array of length ndisc, identifies the ratio between the local derivative in y (normal) position of the grid node by auxiliary coordinate lambda_t 
and the total grid thickness. Used for application of chain rule over finite difference derivatives
us, vs, ws: local velocities (arrays of shape (idisc, jdisc, ndisc)) represented in local coordinate systems
dudz, dwdz: local velocity gradient components in direction perpendicular to inviscid streamlines. Computed using auxiliary coordinates for surface gradient and
renewed once every iteration. network class constructor sets them to 0 as an initial guess, accepting infinite swept wing assumptions

=======
methods
=======

__init__
setcontour
setattachment
calcnorm
propagate
'''