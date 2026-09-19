from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

@dataclass(frozen=True)
class Params:
    hbar: float = 1.0
    m: float = 2.0
    g: float = 1.0
    sigma0: float = 0.60
    z0: float = 3.0
    v0: float = 2.0
    @property
    def p0(self): return self.m*self.v0

def tau(t,p): return p.hbar*np.asarray(t)/(2*p.m*p.sigma0**2)
def sigma_t(t,p): return p.sigma0*np.sqrt(1+tau(t,p)**2)

def classical_center(t,p):
    t=np.asarray(t)
    return p.v0*t, np.zeros_like(t,dtype=float), p.z0-0.5*p.g*t**2

def momentum_expectation(t,p):
    t=np.asarray(t)
    return np.full_like(t,p.p0,dtype=float), np.zeros_like(t,dtype=float), -p.m*p.g*t

def psi(x,y,z,t,p):
    x,y,z,t=map(np.asarray,(x,y,z,t))
    q=1+1j*tau(t,p)
    pref=(2*np.pi*p.sigma0**2)**(-3/4)*q**(-3/2)
    gaussian=np.exp(-((x-p.v0*t)**2+y**2+(z+0.5*p.g*t**2-p.z0)**2)/(4*p.sigma0**2*q))
    plane=np.exp(1j*(p.p0*x-p.p0**2*t/(2*p.m))/p.hbar)
    gravity=np.exp(-1j*(p.m*p.g*t*z+p.m*p.g**2*t**3/6)/p.hbar)
    return pref*gaussian*plane*gravity

def density(x,y,z,t,p):
    s=sigma_t(t,p); xc,_,zc=classical_center(t,p)
    r2=(np.asarray(x)-xc)**2+np.asarray(y)**2+(np.asarray(z)-zc)**2
    return (2*np.pi*s**2)**(-1.5)*np.exp(-r2/(2*s**2))

def energy_expectation(p):
    return p.p0**2/(2*p.m)+3*p.hbar**2/(8*p.m*p.sigma0**2)+p.m*p.g*p.z0

def numerical_normalization(t,p,n=101,nsigma=6.0):
    s=float(sigma_t(t,p)); xc,_,zc=[float(v) for v in classical_center(t,p)]
    x=np.linspace(xc-nsigma*s,xc+nsigma*s,n)
    y=np.linspace(-nsigma*s,nsigma*s,n)
    z=np.linspace(zc-nsigma*s,zc+nsigma*s,n)
    X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
    rho=density(X,Y,Z,t,p)
    return float(np.trapezoid(np.trapezoid(np.trapezoid(rho,z,axis=2),y,axis=1),x,axis=0))

def make_xz_grid(p,tmax,nx=300,nz=240):
    s=float(sigma_t(tmax,p)); x0,_,z0=classical_center(0,p); x1,_,z1=classical_center(tmax,p)
    margin=4.5*s
    x=np.linspace(min(float(x0),float(x1))-margin,max(float(x0),float(x1))+margin,nx)
    z=np.linspace(min(float(z0),float(z1))-margin,max(float(z0),float(z1))+margin,nz)
    X,Z=np.meshgrid(x,z,indexing='xy')
    return x,z,X,Z

def animate_density_xz(p,outpath='density_xz.gif',tmax=2.4,frames=100,fps=25):
    outpath=Path(outpath); x,z,X,Z=make_xz_grid(p,tmax); times=np.linspace(0,tmax,frames)
    rho0=density(X,0,Z,0,p)
    fig,ax=plt.subplots(figsize=(8.4,5.2))
    im=ax.imshow(rho0,origin='lower',extent=[x.min(),x.max(),z.min(),z.max()],aspect='auto',vmin=0,vmax=float(rho0.max()))
    fig.colorbar(im,ax=ax,label=r'$|\psi(x,0,z,t)|^2$')
    dot,=ax.plot([],[],marker='o',linestyle='None',label='packet centre')
    trail,=ax.plot([],[],linestyle='--',lw=1.2,label='classical trajectory')
    ax.set(xlabel='x',ylabel='z'); ax.legend(loc='upper right')
    def update(i):
        t=times[i]; im.set_data(density(X,0,Z,t,p)); xc,_,zc=classical_center(times[:i+1],p)
        dot.set_data([xc[-1]],[zc[-1]]); trail.set_data(xc,zc); ax.set_title(f'Probability density, t={t:.3f}')
        return im,dot,trail
    FuncAnimation(fig,update,frames=frames,interval=1000/fps).save(outpath,writer=PillowWriter(fps=fps)); plt.close(fig); return outpath

def animate_real_psi_xz(p,outpath='real_psi_xz.gif',tmax=2.4,frames=100,fps=25):
    outpath=Path(outpath); x,z,X,Z=make_xz_grid(p,tmax); times=np.linspace(0,tmax,frames)
    f0=np.real(psi(X,0,Z,0,p)); amp=float(np.max(np.abs(f0)))
    fig,ax=plt.subplots(figsize=(8.4,5.2))
    im=ax.imshow(f0,origin='lower',extent=[x.min(),x.max(),z.min(),z.max()],aspect='auto',vmin=-amp,vmax=amp,cmap='RdBu_r')
    fig.colorbar(im,ax=ax,label=r'$\mathrm{Re}\,\psi$'); dot,=ax.plot([],[],marker='o',linestyle='None')
    ax.set(xlabel='x',ylabel='z')
    def update(i):
        t=times[i]; im.set_data(np.real(psi(X,0,Z,t,p))); xc,_,zc=classical_center(t,p); dot.set_data([float(xc)],[float(zc)]); ax.set_title(f'Re(psi), t={t:.3f}'); return im,dot
    FuncAnimation(fig,update,frames=frames,interval=1000/fps).save(outpath,writer=PillowWriter(fps=fps)); plt.close(fig); return outpath

def save_trajectory_spreading_figure(p,outpath='trajectory_spreading.png',tmax=2.4):
    outpath=Path(outpath); t=np.linspace(0,tmax,400); x,_,z=classical_center(t,p); s=sigma_t(t,p)
    fig,ax=plt.subplots(figsize=(7.6,4.8)); ax.plot(x,z,label='packet centre'); ax.fill_between(x,z-s,z+s,alpha=.2,label=r'$\pm\sigma(t)$')
    ax.set(xlabel=r'$\langle x\rangle$',ylabel='z',title='Classical centre with quantum spreading'); ax.legend(); fig.tight_layout(); fig.savefig(outpath,dpi=180); plt.close(fig); return outpath

def save_classical_limit_figure(p,outpath='classical_limit.png',tmax=2.4):
    outpath=Path(outpath); t=np.linspace(0,tmax,400); fig,ax=plt.subplots(figsize=(7.6,4.8))
    for f in (1,5,20):
        pf=replace(p,m=p.m*f); ax.plot(t,sigma_t(t,pf)/pf.sigma0,label=f'm={f:g} m0')
    ax.set(xlabel='t',ylabel=r'$\sigma(t)/\sigma_0$',title='Classical limit: larger mass suppresses spreading'); ax.legend(); fig.tight_layout(); fig.savefig(outpath,dpi=180); plt.close(fig); return outpath

def run_all(p,outdir='outputs',tmax=2.4,frames=100,fps=25):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    print('Parameters:',p); print('<E> =',energy_expectation(p)); print('Norm t=0 =',numerical_normalization(0,p,n=61)); print('Norm t=tmax =',numerical_normalization(tmax,p,n=61))
    paths=[animate_density_xz(p,out/'density_xz.gif',tmax,frames,fps),animate_real_psi_xz(p,out/'real_psi_xz.gif',tmax,frames,fps),save_trajectory_spreading_figure(p,out/'trajectory_spreading.png',tmax),save_classical_limit_figure(p,out/'classical_limit.png',tmax)]
    for q in paths: print('created',q)

def main():
    a=argparse.ArgumentParser()
    for name,typ,default in [('outdir',str,'outputs'),('tmax',float,2.4),('frames',int,100),('fps',int,25),('m',float,2.0),('g',float,1.0),('sigma0',float,.6),('z0',float,3.0),('v0',float,2.0),('hbar',float,1.0)]:
        a.add_argument('--'+name,type=typ,default=default)
    q=a.parse_args(); p=Params(q.hbar,q.m,q.g,q.sigma0,q.z0,q.v0); run_all(p,q.outdir,q.tmax,q.frames,q.fps)

if __name__=='__main__':
    main()
