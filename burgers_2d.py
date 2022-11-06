# Solving burgers equation in 1D by FD

# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# variables and discritization parameters
nt=500000
nx=2000
nz=1000
c=1
Ra = 100000.
Pr = 1.0
const1 = np.sqrt(Pr/Ra)
const2 = 1./np.sqrt(Pr*Ra)
dt = 2.e-5
dx=1.0/(nx-1)
dz=1.0/(nz-1)

x = np.linspace(0, 1, nx)
z = np.linspace(0, 1, nz)

u = np.zeros((nx,nz))
v = np.zeros((nx,nz))
t = np.zeros((nx,nz))

un = np.zeros((nx,nz))
vn = np.zeros((nx,nz))
tn = np.zeros((nx,nz))

# assigning initial conditions
seed=int(time.time())
np.random.seed(seed) #pick a random seed
for i in range(1,nz-1):
    for j in range(1,nx-1):
        u[i,j]=(2.*np.random.rand()-1.)*0.03 +0.01*np.sin(2.*np.pi*i*dz)
        v[i,j]=(2.*np.random.rand()-1.)*0.03 +0.01*np.sin(2.*np.pi*i*dz)
        t[i,j]=(2.*np.random.rand()-1.)*0.03 +0.01*np.sin(2.*np.pi*i*dz)

plt.plot(z,u)
plt.plot(z,v)
plt.plot(z,np.zeros(len(z)))
plt.show()


# loop across number of time steps
dt_over_dz = (dt/dz)
dt_over_dz2 = dt/(dz**2)
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    """
    # The following algorithm is slow
    for i in range(1,nz-1):
        u[i] = un[i] - (un[i] * dt_over_dz * (un[i] - un[i-1])) + (const1*dt_over_dz2)*(un[i+1]-2*un[i]+un[i-1]) + dt*vn[i] 
        v[i] = vn[i] - (un[i] * dt_over_dz * (vn[i] - vn[i-1])) + (const2*dt_over_dz2)*(vn[i+1]-2*vn[i]+vn[i-1]) + dt*un[i]
        """
    # Much faster algorithm
    unm=np.roll(u,1)
    unp=np.roll(u,-1)
    vnm=np.roll(v,1)
    vnp=np.roll(v,-1)
    u = un - (un * dt_over_dz * (un - unm)) + (const1*dt_over_dz2)*(unp-2*un+unm) + (dt*vn) 
    v = vn - (un * dt_over_dz * (vn - vnm)) + (const2*dt_over_dz2)*(vnp-2*vn+vnm) + (dt*un)
    # Set zero mode to zero
    u = u-np.mean(u)*np.ones(nz)
    v = v-np.mean(v)*np.ones(nz)
    # Velocity and temperature boundary conditions
    u[0] = 0
    u[-1]= 0
    v[0] = 0
    v[-1] = 0
# plot 
plt.plot(z,u)
plt.plot(z,v)
plt.plot(z,np.zeros(len(z)))
plt.show()
