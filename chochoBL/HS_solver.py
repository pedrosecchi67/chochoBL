#Check Cebeci, Smith et. Al for reference
#File containing classes, methods and info on solution of Hansen-Yohrner equations for normalized boundary layer shape parameters

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