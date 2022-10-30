# Solving burgers equation in 1D by FD

# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# variables and discritization parameters
nt=500000
nx=1000
c=1
Ra = 100000.
Pr = 1.0
const1 = np.sqrt(Pr/Ra)
const2 = 1./np.sqrt(Pr*Ra)
dt = 2.e-5

dx=1.0/(nx-1)

x = np.linspace(0, 1, nx)

u = np.zeros(nx)
v = np.zeros(nx)
un = np.zeros(nx)
vn = np.zeros(nx)
# assigning initial conditions
seed=int(time.time())
np.random.seed(seed) #pick a random seed
for i in range(1,nx-1):
    u[i]=(2.*np.random.rand()-1.)*0.03 +0.01*np.sin(2.*np.pi*i*dx)
    v[i]=(2.*np.random.rand()-1.)*0.03 +0.01*np.sin(2.*np.pi*i*dx)
 

plt.plot(x,u)
plt.plot(x,v)
plt.plot(x,np.zeros(len(x)))
plt.show()


# loop across number of time steps
dt_over_dx = (dt/dx)
dt_over_dx2 = dt/(dx**2)
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    """
    # The following algorithm is slow
    for i in range(1,nx-1):
        u[i] = un[i] - (un[i] * dt_over_dx * (un[i] - un[i-1])) + (const1*dt_over_dx2)*(un[i+1]-2*un[i]+un[i-1]) + dt*vn[i] 
        v[i] = vn[i] - (un[i] * dt_over_dx * (vn[i] - vn[i-1])) + (const2*dt_over_dx2)*(vn[i+1]-2*vn[i]+vn[i-1]) + dt*un[i]
        """
    # Much faster algorithm
    unm=np.roll(u,1)
    unp=np.roll(u,-1)
    vnm=np.roll(v,1)
    vnp=np.roll(v,-1)
    u = un - (un * dt_over_dx * (un - unm)) + (const1*dt_over_dx2)*(unp-2*un+unm) + (dt*vn) 
    v = vn - (un * dt_over_dx * (vn - vnm)) + (const2*dt_over_dx2)*(vnp-2*vn+vnm) + (dt*un)
    # Set zero mode to zero
    u = u-np.mean(u)*np.ones(nx)
    v = v-np.mean(v)*np.ones(nx)
    # Velocity and temperature boundary conditions
    u[0] = 0
    u[-1]= 0
    v[0] = 0
    v[-1] = 0
# plot 
plt.plot(x,u)
plt.plot(x,v)
plt.plot(x,np.zeros(len(x)))
plt.show()
