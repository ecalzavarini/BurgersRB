#!/usr/bin/env python
import numpy as np
from math import *
import os 
from shutil import copyfile
##################################################################
# read parameters from file
with open('param.in', 'r') as f:
    param_in = f.read().split()
f.closed

N=int(param_in[param_in.index('N')+1])
print("N = ", N)

dt=float(param_in[param_in.index('dt')+1])
print("dt = ", dt)

dump_time=float(param_in[param_in.index('dump_time')+1])
print("dump_time = ", dump_time)

end_t=float(param_in[param_in.index('end_t')+1])
print("end_t = ", end_t)

Ra=float(param_in[param_in.index('Ra')+1])
print("Ra = ", Ra)

Pr=float(param_in[param_in.index('Pr')+1])
print("Pr = ", Pr)

##redefine dt
dt_input = dt
par_k=2.*np.pi
par_lambda =-(par_k**2.)*(Pr+1)/2. + np.sqrt( ((Pr+1.)**2.)*(par_k**4.) + 4.*Pr*(Ra-par_k**4.) )/2.
par_lambda = par_lambda/sqrt(Pr*Ra)
dt = float(0.01 * 1.0/par_lambda)
print("redefined dt = ", dt)
def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))
dt = round_to_1(dt)
print("rounded dt = ", dt)
if dt <0 or dt > dt_input:
    dt = dt_input
print("used dt = ", dt)


#Prepare some file names
dirname='RUN_Ra'+str(Ra)+'_Pr'+str(Pr)+'_N'+str(N)+'_dt'+str(dt)+'/'
print(dirname)
os.makedirs(dirname)
copyfile("param.in",dirname+"param.in")
fname='_Ra'+str(Ra)+'_Pr'+str(Pr)+'_N'+str(N)+'_dt'+str(dt)+'.dat'

with open(dirname+'param'+fname, 'a+') as f:    
    print('Ra',str(Ra),file=f)
    print('Pr',str(Pr),file=f)
    print('N',str(N),file=f)
    print('dt',str(dt),file=f)    
f.closed   

###################################################################
# Define global fields

dx = 1.0/float(N)

NC = N//2 + 1 # this depends onthe type of Fourier transfrom adopted

max_radius =  2.0*np.pi*float(N)/3.

#define arrays  and initialize some of them
wc = np.zeros(NC,complex)
wr = np.zeros(N,float)

nlc = np.zeros(NC,complex)
nlr = np.zeros(N,float)

old_nlc = np.zeros(NC,complex) 

fc = np.zeros(NC,complex)
fr = np.zeros(N,float) 

k = 2.0*np.pi*np.fft.rfftfreq(N, 1./N)
kamp = np.zeros(NC,float)
exp_fac = np.zeros(NC,float)
#print(k.dtype)
#print(wr.dtype)
#print(wc.dtype)

for i in range(NC):
    k2=k[i]*k[i]
    exp_fac[i] = exp(-sqrt(Pr/Ra)*dt*k2)
    kamp[i] = sqrt(k2)

##### Definitions for temperature
tc = np.zeros(NC,complex)
tr = np.zeros(N,float)

tnlc = np.zeros(NC,complex)
tnlr = np.zeros(N,float)

old_tnlc = np.zeros(NC,complex) 

sc = np.zeros(NC,complex)
sr = np.zeros(N,float) 

exp_tfac = np.zeros(NC,float)

for i in range(NC):
    k2=k[i]*k[i]
    exp_tfac[i] = exp(-sqrt(1./(Pr*Ra))*dt*k2)
###################################################################
# compute the non linear term ( u \cdot \grad) u 
def compute_nl():
    global nlc, nlr, wc, wr, k, N, kamp, max_radius
    global tnlc,tnlr,tc,tr

# compute gradient 
    for i in range(NC):
        if kamp[i] >  max_radius:
            nlc[i]=0.0+0.0j  #this is dealiasing
            tnlc[i]=0.0+0.0j  #this is dealiasing
        else:
            nlc[i]=1j*k[i]*wc[i]
            tnlc[i]=1j*k[i]*tc[i]
            
# go to physical space k->r
    nlr=np.fft.irfft(nlc)
    wr=np.fft.irfft(wc)
    tnlr=np.fft.irfft(tnlc)
    tr=np.fft.irfft(tc) # not needed
# product and change sign
    nlr=-wr*nlr
    tnlr=-wr*tnlr
# go to fourier space again r->k
    nlc=np.fft.rfft(nlr)    
    tnlc=np.fft.rfft(tnlr)
####################################################################
def add_forcing():
    global fr,fc,nlc,N,Pr,Ra
    global sr,sc,tnlc
    fc = np.copy(tc)
    sc = np.copy(wc)
    #noise
    #fc +=  + np.ones(NC,complex)*(2.*np.random.rand()-1.)
    nlc += fc
    tnlc += sc
####################################################################
# advance in time
def time_stepping(it):
    global wc, nlc , old_nlc, dt, exp_fac
    global tc, tnlc, old_tnlc, exp_tfac
    
    if it==0:
        a,b=1.0,0.0
    else:
        a,b=1.5,0.5  
# AB2 or EULER
    wc = (wc+dt*(a* nlc -b*old_nlc*exp_fac))*exp_fac
    tc = (tc+dt*(a*tnlc -b*old_tnlc*exp_tfac))*exp_tfac    
# copy old right hand side term
    old_nlc = np.copy(nlc)
    old_tnlc = np.copy(tnlc)
###############################################################
def zero_mode_at_zero():
    global wc,tc 
    wc[0]=0.0+0.0j
    tc[0]=0.0+0.0j
###############################################################
def make_odd():
    global wc,tc 
    wc.real=0.0
    tc.real=0.0
###############################################################
def add_penalization():
    global wr,wc,fr,fc,nlc,N,Pr,Ra
    global tr,tc,sr,sc,tnlc
    # in spectral space this look like a constant
    K=N
    nlc -= K*np.ones(NC)*wr[0]
    tnlc -= K*np.ones(NC)*tr[0]
###############################################################
def write_field(it):
    global N,dx
    global wc,wr,tr,tc
    global fname
    wr = np.fft.irfft(wc)
    tr = np.fft.irfft(tc)
    with open(dirname+'field'+fname, 'a+') as f:
        print("# it = ",it,file=f)
        for i in range(N):
            print(dx*i,wr[i],tr[i],wr[i]*tr[i],file=f)
    f.closed    
###############################################################
def write_averages(it):
    global N,dt
    global wr,tr
    global fname
    with open(dirname+'averages'+fname, 'a+') as f:    
        print(it*dt,np.sum(wr)/N,np.sum(np.abs(wr))/N,np.sum(tr)/N,np.sum(np.abs(tr))/N,np.sum(wr*tr)/N,file=f)
    f.closed
###############################################################
def write_final_averages():
    global N,dt,Ra,Pr
    global wr,tr
    global fname
    with open(dirname+'final_averages'+fname, 'w+') as f:    
        print(Ra,Pr,np.sum(wr)/N,np.sum(np.abs(wr))/N,np.sum(tr)/N,np.sum(np.abs(tr))/N,np.sum(wr*tr)/N,file=f)
    f.closed
###############################################################
def write_final_field(it):
    global N,dx
    global wc,wr,tr,tc
    global fname
    wr = np.fft.irfft(wc)
    tr = np.fft.irfft(tc)
    with open(dirname+'final_field'+fname, 'w+') as f:
        print("# it = ",it,file=f)
        wr_z= np.gradient(wr)*N
        wr_zz = np.gradient(wr_z)*N
        for i in range(N):
            print(dx*i,wr[i],tr[i],wr[i]*tr[i],wr_z[i],wr_zz[i],file=f)
    f.closed        
###############################################################
def write_spectra(it):
    global N
    global wc,tc
    global fname
    with open(dirname+'spectra'+fname, 'a+') as f:
        print("# it = ",it,file=f)
        for i in range(NC):
            print(np.abs(k[i]), (np.abs(wc[i]))**2., (np.abs(tc[i]))**2., wc.real[i], wc.imag[i], tc.real[i], tc.imag[i], file=f)
    f.closed       
##############################################################
# initialize velocity
def initialize_fields():
    global wr, wc, tr,tc,N
    np.random.seed()
    print(np.random.rand())
    for i in range(N):
        #wr[i]=np.sin(2*np.pi*i*dx)
        #tr[i]=np.sin(2*np.pi*i*dx)
        wr[i]=(2.*np.random.rand()-1.)*0.00001
        tr[i]=(2.*np.random.rand()-1.)*0.00001
    wc=np.fft.rfft(wr)
    tc=np.fft.rfft(tr)
    with open(dirname+'initial.dat', 'w') as f:    
        for i in range(N):
            print(dx*i,wr[i],tr[i],file=f)            
    f.closed
    with open(dirname+'initial-complex.dat', 'w') as f:    
        for i in range(NC):
            print(k[i],wc[i].real,wc[i].imag,tc[i].real,tc[i].imag,file=f)            
    f.closed     
###############################################################  
# check convergence (flow steadiness) for stopping the simulation
tot_ene_old = 0
def check_steady():
    global wr
    global tot_ene_old
    tot_ene_now = np.sum(np.abs(wr))
    if np.abs(tot_ene_now - tot_ene_old)<1.e-14:
        tot_ene_old = tot_ene_now
        return 1
    else:
        tot_ene_old = tot_ene_now
        return 0
############################################################### 
# MAIN 
###############################################################
# Here the simulation starts
initialize_fields()

dump_it = int(dump_time/dt)
print(dump_it)

# main loop on time
#for it in range(0, int(end_t/dt)):
it = val = 0
while it < int(end_t/dt) and val == 0:

#compute real velocity field    
    if it%dump_it == 0:
        write_field(it)
        write_spectra(it)
    if it%10 == 0:
        write_averages(it)
#compute non linear term
    compute_nl()
#add forcing    
    add_forcing()
    #add_penalization()
#advance in time
    time_stepping(it)
    zero_mode_at_zero()
    make_odd()
#diagnostic and output
    if it%10 == 0:
        print("it = ",it)
#check convergence
    if it%dump_it == 0:
        val=check_steady()
    if val == 1:
        print("run converged at it =",it)
    it = it +1
# final writes
write_final_averages()
write_final_field(it)
################################################################

