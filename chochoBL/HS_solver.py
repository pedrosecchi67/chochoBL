#Check Cebeci, Smith et. Al for reference
#File containing classes, methods and info on solution of Falkner-Skan-Crooke equations for normalized boundary layer shape parameters
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smsc
import scipy.optimize as sopt
import scipy.interpolate as sinterp
from math import *
import pickle

def FSC_step(b=1.0, bp=0.0, f=0.0, fp=0.0, fpp=0.0, g=0.0, gp=0.0, gpp=0.0, eta=0.0, deta=1e-3, m=0):
    #function to consider a finite difference step in solving Hansen-Yohrner's equations
    #obtaining (bf'')' and (bg'')':
    #applying equations
    bfp=-((m+1)*f*fpp/2+m*(1.0-fp**2))
    bgp=-((m+1)*f*gpp/2)
    #deducing fppp and gppp based on previous results and derivative of b
    fppp=(bfp-bp*fpp)/b
    gppp=(bgp-bp*gpp)/b
    return f+deta*fp, fp+deta*fpp, fpp+deta*fppp, g+deta*gp, gp+deta*gpp, gpp+deta*gppp

def FSC_IV_solve(nstep, etas, bs, bps, fpp_guess=0.0, gpp_guess=0.0, m=0):
    #solve equations with finite difference scheme for given initial value guess
    fps=np.zeros(nstep); fpps=np.zeros(nstep); gps=np.zeros(nstep); gpps=np.zeros(nstep)
    f=0.0; g=0.0; fp=0.0; gp=0.0; fpp=fpp_guess; gpp=gpp_guess; fppp=0.0; gppp=0.0
    for i in range(nstep-1):
        fps[i]=fp; gps[i]=gp; fpps[i]=fpp; gpps[i]=gpp
        f, fp, fpp, g, gp, gpp=FSC_step(b=bs[i], bp=bps[i], f=f, g=g, fp=fp, gp=gp, fpp=fpp, gpp=gpp, eta=etas[i], deta=etas[i+1]-etas[i], m=m)
    fps[-1]=fp; gps[-1]=gp; fpps[-1]=fpp; gpps[-1]=gpp
    return fps, fpps, gps, gpps

def FSC_examine(nstep, etas, bs, bps, fpp_guess=0.0, gpp_guess=0.0, n=0, m=0, BA=0.0, plot=True): #examine residual for shooting method solution
    fps, _, gps, _=FSC_IV_solve(nstep, etas, bs, bps, fpp_guess=fpp_guess, gpp_guess=gpp_guess, m=m)
    if plot:
        plt.plot(fps, etas, label='$f\'(\eta)$')
        plt.plot(gps, etas, label='$g\'(\eta)$')
        plt.xlabel('$f(\eta)$')
        plt.ylabel('$\eta$')
        plt.legend()
        plt.show()
    return (1.0-fps[-1])**2+(1.0-gps[-1])**2

def FSC_solution(nstep=100, eta_max=10.0, b=lambda eta: np.ones(len(eta)), b_dh=1e-4, strategy=lambda x: x, fpp_guess=0.0, gpp_guess=0.0, \
    m=0.0, tol=1e-4):
    #execute shooting method solution
    etas=strategy(np.linspace(0.0, 1.0, nstep))*eta_max
    bs=b(etas)
    bps=(b(etas+b_dh)-b(etas-b_dh))/(2*b_dh)
    x=np.array([fpp_guess, gpp_guess]) #initial guess for initial value conditions
    #if np.all(x==0.0):
    #    x=np.array([m, (n+1)*BA/2])
    foo=lambda g: FSC_examine(nstep, etas, bs, bps, fpp_guess=g[0], gpp_guess=g[1], m=m, plot=False)
    sln=sopt.minimize(method='Nelder-Mead', fun=foo, x0=x, tol=tol)
    return not sln.success, sln.x

def FSC_abaqus_element(nstep=100, eta_max=10.0, b=lambda eta: np.ones(len(eta)), b_dh=1e-4, strategy=lambda x: x, fpp_guess=0.0, gpp_guess=0.0, \
    m=0.0, tol=1e-4):
    succ, x=FSC_solution(nstep=nstep, eta_max=eta_max, b=b, b_dh=b_dh, strategy=strategy, fpp_guess=fpp_guess, gpp_guess=gpp_guess, \
        m=m, tol=tol)
    abaqdict={}
    if not succ:
        etas=strategy(np.linspace(0.0, 1.0, nstep))*eta_max
        bs=b(etas)
        bps=(b(etas+b_dh)-b(etas-b_dh))/(2*b_dh)
        fps, fpps, gps, gpps=FSC_IV_solve(nstep=nstep, etas=etas, bs=bs, bps=bps, fpp_guess=x[0], gpp_guess=x[1], m=m)
        abaqdict['Dx']=np.trapz(1.0-fps, etas)
        abaqdict['Dz']=np.trapz(1.0-gps, etas)
        abaqdict['Thxx']=np.trapz((1.0-fps)*fps, etas)
        abaqdict['Thxz']=np.trapz((1.0-fps)*gps, etas)
        abaqdict['Thzx']=np.trapz((1.0-gps)*fps, etas)
        abaqdict['Thzz']=np.trapz((1.0-gps)*gps, etas)
        abaqdict['Lx']=abaqdict['Thxx']**2*m
        abaqdict['fpp']=fpps[0]
        abaqdict['gpp']=gpps[0]
        d1=etas[np.abs(fps-1.0)<0.01]
        d2=etas[np.abs(gps-1.0)<0.01]
        isvalid=False
        posvals=[]
        if len(d1)!=0:
            posvals+=[d1[0]]
            isvalid=True
        if len(d2)!=0:
            posvals+=[d2[0]]
            isvalid=True
        if not isvalid:
            raise Exception('Invalid solution')
        abaqdict['delta']=max(posvals)
        plt.plot(fps, etas)
        plt.plot(gps, etas)
        plt.show()
    return succ, abaqdict

#emax=10.0
#print(FSC_abaqus_element(nstep=2000, eta_max=emax, m=-0.09, fpp_guess=0.0, gpp_guess=0.2, tol=1e-4))

Spalding_yp=lambda Ups: Ups+0.1108*(np.exp(0.4*Ups)-1.0-0.4*Ups-0.5*(0.4*Ups)**2-(0.4*Ups)**3/6)
base_Ups=np.linspace(0.01, 20.0, 1000)
base_yps=Spalding_yp(base_Ups)
Spalding_wall_function=sinterp.UnivariateSpline(base_yps, base_yps/base_Ups, ext=0)
#plt.plot(base_yps, base_Ups)
#plt.xscale('log')
#plt.show()
del Spalding_yp, base_Ups, base_yps

def Von_Karman_wall_function(yps): #takes: y+ normalized coordinate
    #returns: b=(1+nu_t/nu)
    laminar_subregion=yps<11.25
    turbulent_subregion=np.logical_not(laminar_subregion)
    bs=np.zeros(len(yps))
    bs[laminar_subregion]=1.0
    #log law: U+=log(E*y+)/kappa
    #kappa: 0.4187; E: 9.793
    bs[turbulent_subregion]=(yps[turbulent_subregion]*0.4187)/np.log(9.793*yps[turbulent_subregion])
    return bs

class quasi3d_abaqus:
    def __init__(self, nstep=1000, etamax_SF=2.0, disc_m=20, mlims=[-0.09, 1.0], mstrategy=lambda x: (np.exp(x)-1.0)/(np.exp(1)-1.0), disc_etastar=20, etastar_lims=[0.0, 50.0], \
        etastar_strategy=lambda x : (np.exp(x)-1.0)/(np.exp(1)-1.0), eta_strategy=lambda x : x, tol=1e-4, b_dh=1e-4, wall_function=Von_Karman_wall_function):
        self.ms=np.interp(mstrategy(np.linspace(0.0, 1.0, disc_m)), np.array([0.0, 1.0]), np.array(mlims))
        self.etastar=np.interp(etastar_strategy(np.linspace(0.0, 1.0, disc_etastar)), np.array([0.0, 1.0]), np.array(etastar_lims))
        print(self.ms)
        Dxb=np.zeros((disc_m, disc_etastar))
        Dzb=np.zeros((disc_m, disc_etastar))
        Thxxb=np.zeros((disc_m, disc_etastar))
        Thxzb=np.zeros((disc_m, disc_etastar))
        Thzxb=np.zeros((disc_m, disc_etastar))
        Thzzb=np.zeros((disc_m, disc_etastar))
        Lxb=np.zeros((disc_m, disc_etastar))
        fppb=np.zeros((disc_m, disc_etastar))
        gppb=np.zeros((disc_m, disc_etastar))
        deltab=np.zeros((disc_m, disc_etastar))
        for etast, j in zip(self.etastar, range(disc_etastar)):
            for m, i in zip(self.ms, range(disc_m)):
                if j!=0 and i==0:
                    emax=deltab[i, j-1]*etamax_SF
                    fguess, gguess=fppb[i, j-1], gppb[i, j-1]
                elif i==0:
                    emax=6.0*etamax_SF
                    fguess, gguess=0.0, 0.2
                else:
                    fguess, gguess=fppb[i-1, j], gppb[i-1, j]
                    emax=deltab[i-1, j]*etamax_SF
                iserror, abaqdict=FSC_abaqus_element(m=m, nstep=nstep, eta_max=emax, fpp_guess=fguess, gpp_guess=gguess, tol=tol, strategy=eta_strategy, b_dh=b_dh, \
                    b=lambda eta: wall_function(etast*eta))
                if iserror:
                    raise Exception('Nelder-Mead shooting method failed')
                print(abaqdict)
                Dxb[i, j]=abaqdict['Dx']
                Dzb[i, j]=abaqdict['Dz']
                Thxxb[i, j]=abaqdict['Thxx']
                Thxzb[i, j]=abaqdict['Thxz']
                Thzxb[i, j]=abaqdict['Thzx']
                Thzzb[i, j]=abaqdict['Thzz']
                Lxb[i, j]=abaqdict['Lx']
                fppb[i, j]=abaqdict['fpp']
                gppb[i, j]=abaqdict['gpp']
                deltab[i, j]=abaqdict['delta']
        self.Dx=Dxb
        self.Dz=Dzb
        self.Thxx=Thxxb
        self.Thxz=Thxzb
        self.Thzx=Thzxb
        self.Thzz=Thzzb
        self.fpp=fppb
        self.gpp=gppb
        self.Lx=Lxb

def abaqus_dump(abaq, fname):
    fil=open(fname, 'wb')
    pickle.dump(abaq, fil)
    fil.close()

def abaqus_load(fname):
    fil=open(fname, 'rb')
    fil.close()
    return abaq

#definition for an abaqus: capable of returning all properties and ratios for
#BL on response to Lx and delta-star input