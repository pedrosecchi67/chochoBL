import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.interpolate as sinterp
import time as tm

class abaqus:
    def __init__(self, Alphas, Reths, Cf_mat, H_mat):
        self.Cf_rule=sinterp.RectBivariateSpline(Alphas, Reths, Cf_mat)
        self.H_rule=sinterp.RectBivariateSpline(Alphas, Reths, H_mat)
    def __call__(self, Alpha, R):
        return self.Cf_rule(Alpha, R), self.H_rule(Alpha, R)

def White_solve(Alpha, R, Cf_guess=1.5, H_guess=1.0, locus_A=6.7, locus_B=0.75):
    residual=lambda t: (t[0]-0.3*np.exp(-1.33*t[1])/((np.log10(R))**(1.74+0.31*t[1])), ((t[1]-1.0)/(locus_A*t[1]))**2-t[0]/2+locus_B*t[1]*Alpha)
    return sopt.fsolve(residual, (Cf_guess, H_guess))

def White_solve_mat(Alphas, Rs, Cf_guess=1.5, H_guess=1.0, locus_A=6.7, locus_B=0.75, plot=False):
    Cf_mat=np.zeros((len(Alphas), len(Rs)))
    H_mat=np.zeros((len(Alphas), len(Rs)))
    Cg=Cf_guess; Hg=H_guess
    for i in range(len(Alphas)):
        if i!=0:
            Cg=Cf_mat[i-1, 0]; Hg=H_mat[i-1, 0]
        for j in range(len(Rs)):
            Cf_mat[i, j], H_mat[i, j]=White_solve(Alphas[i], Rs[j], locus_A=locus_A, locus_B=locus_B, Cf_guess=Cg, H_guess=Hg)
            Cg=Cf_mat[i, j]; Hg=H_mat[i, j]
    if plot:
        fig=plt.figure()
        ax=plt.axes(projection='3d')
        Ag, Rg=np.meshgrid(Alphas, Rs)
        ax.plot_surface(Ag, Rg, Cf_mat)
        plt.title('$C_f$')
        plt.xlabel('$\Gamma$')
        plt.ylabel('$Re_{\Theta}$')
        plt.show()
        fig=plt.figure()
        ax=plt.axes(projection='3d')
        ax.plot_surface(Ag, Rg, H_mat)
        plt.title('$H$')
        plt.xlabel('$\Gamma$')
        plt.ylabel('$Re_{\Theta}$')
        plt.show()
    return Cf_mat, H_mat

As=np.linspace(0.0, 1.5, 50)
Rs=np.linspace(1.1, 2.0, 50)
Cfs, Hs=White_solve_mat(Alphas=As, Rs=Rs, \
    plot=True)
abaq=abaqus(As, Rs, Cfs, Hs)
t=tm.time()
Cf, H=abaq(0.4, 1.3)
print(tm.time()-t)