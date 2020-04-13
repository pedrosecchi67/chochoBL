#Check Cebeci, Smith et. Al for reference
#File containing classes, methods and info on solution of Hansen-Yohrner equations for normalized boundary layer shape parameters
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smsc
from math import *

def HS_step(b=1.0, bp=0.0, f=0.0, fp=0.0, fpp=0.0, g=0.0, gp=0.0, gpp=0.0, eta=0.0, deta=1e-3, n=0, m=0, BA=0.0):
    #function to consider a finite difference step in solving Hansen-Yohrner's equations
    #obtaining (bf'')' and (bg'')':
    #applying equations
    bfp=-((m+1)*f*fpp/2+m*(1.0-fp**2)+BA*((n+2)*g*fpp/2+n*(1.0-fp*gp)))
    bgp=-((m+1)*f*gpp/2+(m-1)*(1.0-fp*gp)+BA*((n+2)*g*gpp/2+(n+1)*(1.0-gp**2)))
    #deducing fppp and gppp based on previous results and derivative of b
    fppp=(bfp-bp*fpp)/b
    gppp=(bgp-bp*gpp)/b
    return f+deta*fp, fp+deta*fpp, fpp+deta*fppp, g+deta*gp, gp+deta*gpp, gpp+deta*gppp

def HS_IV_solve(nstep, etas, bs, bps, fpp_guess=0.0, gpp_guess=0.0, n=0, m=0, BA=0.0):
    #solve equations with finite difference scheme for given initial value guess
    fps=np.zeros(nstep); fpps=np.zeros(nstep); gps=np.zeros(nstep); gpps=np.zeros(nstep)
    f=0.0; g=0.0; fp=0.0; gp=0.0; fpp=fpp_guess; gpp=gpp_guess; fppp=0.0; gppp=0.0
    for i in range(nstep-1):
        fps[i]=fp; gps[i]=gp; fpps[i]=fpp; gpps[i]=gpp
        f, fp, fpp, g, gp, gpp=HS_step(b=bs[i], bp=bps[i], f=f, g=g, fp=fp, gp=gp, fpp=fpp, gpp=gpp, eta=etas[i], deta=etas[i+1]-etas[i], n=n, m=m, BA=BA)
    fps[-1]=fp; gps[-1]=gp; fpps[-1]=fpp; gpps[-1]=gpp
    return fps, fpps, gps, gpps

def HS_examine(nstep, etas, bs, bps, fpp_guess=0.0, gpp_guess=0.0, n=0, m=0, BA=0.0, plot=True): #examine residual for shooting method solution
    fps, _, gps, _=HS_IV_solve(nstep, etas, bs, bps, fpp_guess=fpp_guess, gpp_guess=gpp_guess, n=n, m=m, BA=BA)
    if plot:
        plt.plot(fps, etas, label='$f\'(\eta)$')
        plt.plot(gps, etas, label='$g\'(\eta)$')
        plt.xlabel('$f(\eta)$')
        plt.ylabel('$\eta$')
        plt.legend()
        plt.show()
    return (1.0-fps[-1])**2+(1.0-gps[-1])**2

def HS_solution(nstep=100, eta_max=10.0, niter=10, b=lambda eta: np.ones(len(eta)), b_dh=1e-4, strategy=lambda x: x, fpp_guess=0.0, gpp_guess=0.0, \
    fguess_step=0.1, gguess_step=0.1, newton_relaxation=0.1, m=0.0, n=0.0, BA=0.0, \
        echo=True, findiff_relax=True):
    #execute shooting method solution
    etas=strategy(np.linspace(0.0, 1.0, nstep))*eta_max
    bs=b(etas)
    bps=(b(etas+b_dh)-b(etas-b_dh))/(2*b_dh)
    h_f=fguess_step; h_g=gguess_step
    x=np.array([fpp_guess, gpp_guess]) #initial guess for initial value conditions
    if np.all(x==0.0):
        x=np.array([m, (n+1)*BA/2])
    gradR=np.zeros(2)
    Rcentral_old=1.0
    if echo:
        print('==========Shooting Method for Hansen-Yohrner equations=======')
    for i in range(1, niter+1):
        Rcentral=HS_examine(nstep, etas, bs, bps, fpp_guess=x[0], gpp_guess=x[1], n=n, m=m, BA=BA, plot=True)
        gradR[0]=(HS_examine(nstep, etas, bs, bps, fpp_guess=x[0]+h_f, gpp_guess=x[1], n=n, m=m, BA=BA, plot=False)-\
            HS_examine(nstep, etas, bs, bps, fpp_guess=x[0]-h_f, gpp_guess=x[1], n=n, m=m, BA=BA, plot=False))/(2*h_f)
        gradR[1]=(HS_examine(nstep, etas, bs, bps, fpp_guess=x[0], gpp_guess=x[1]+h_g, n=n, m=m, BA=BA, plot=False)-\
            HS_examine(nstep, etas, bs, bps, fpp_guess=x[0], gpp_guess=x[1]-h_g, n=n, m=m, BA=BA, plot=False))/(2*h_g)
        if echo:
            print('==============%dth iteration==============' % (i))
            print('%8s %8s %8s %20s %20s' % ('R', 'h_f', 'h_g', '(fpp, gpp)', 'gradR(x)'))
            print('%8f %8f %8f (%8f, %8f) (%8f, %8f)' % (Rcentral, h_f, h_g, x[0], x[1], gradR[0], gradR[1]))
        if i!=0 and findiff_relax:
            h_f*=Rcentral/Rcentral_old
            h_g*=Rcentral/Rcentral_old
        Rcentral_old=Rcentral
        x-=newton_relaxation*gradR

#HS_solution(fpp_guess=1.0, gpp_guess=0.0, niter=20, m=1.0, newton_relaxation=0.1, BA=0.0)