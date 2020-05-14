import chochoBL as cho
import numpy as np

#loading data
velsx=np.loadtxt('N0012AR10_5deg_vels0.dat')
velsy=np.loadtxt('N0012AR10_5deg_vels1.dat')
velsz=np.loadtxt('N0012AR10_5deg_vels2.dat')
positsx=np.loadtxt('N0012AR10_5deg_posits0.dat')
positsy=np.loadtxt('N0012AR10_5deg_posits1.dat')
positsz=np.loadtxt('N0012AR10_5deg_posits2.dat')

#reshaping
xdisc=np.size(velsx, 0)
ydisc=np.size(velsx, 1)

vels=np.zeros((xdisc, ydisc, 3))
posits=np.zeros((xdisc, ydisc, 3))

vels[:, :, 0]=velsx
vels[:, :, 1]=velsy
vels[:, :, 2]=velsz
posits[:, :, 0]=positsx
posits[:, :, 1]=positsy
posits[:, :, 2]=positsz

del velsx, velsy, velsz, positsx, positsy, positsz

msh=cho.mesh(posits=posits, vels=vels)

print(msh._identify_starting_points())