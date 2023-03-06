import numpy as np
from generate_data.parametric_pde_find import *
from scipy.integrate import odeint
from numpy.fft import fft, ifft, fftfreq
from time import time

def parametric_burgers_rhs(u, t, params):
    k,a,b,c = params
    deriv = a*(1+c*np.sin(t))*u*ifft(1j*k*fft(u)) + b*ifft(-k**2*fft(u))
    return np.real(deriv)

# Set size of grid
n = 512
m = 256

# Set up grid
x = np.linspace(-8,8,n+1)[:-1];   dx = x[1]-x[0]
t = np.linspace(0,10,m);          dt = t[1]-t[0]
k = 2*np.pi*fftfreq(n, d = dx)

# Initial condition
u0 = np.exp(-(x+1)**2)

# Solve with time dependent uu_x term
params = (k, -1, 0.1, 0.25)
u = odeint(parametric_burgers_rhs, u0, t, args=(params,)).T

u_xx_true = 0.1*np.ones(m)
uu_x_true = -1*(1+0.25*np.sin(t))

Ut, Theta, rhs_des = build_linear_system(u, dt, dx, D=4, P=3, time_diff = 'FD', space_diff = 'FD')

Theta_grouped = [np.real(Theta[j*n:(j+1)*n,:]) for j in range(m)]
Ut_grouped = [np.real(Ut[j*n:(j+1)*n]) for j in range(m)]

u = np.stack([Tg[:,1] for Tg in Theta_grouped], axis=0)
u_x = np.stack([Tg[:,4] for Tg in Theta_grouped], axis=0)
u_xx = np.stack([Tg[:,8] for Tg in Theta_grouped], axis=0)

u_t = np.stack([Ug[:,0] for Ug in Ut_grouped], axis=0)

in_data = np.stack([u, u_x, u_xx], axis=2).astype('float32')
out_data = np.stack([u_t], axis=2).astype('float32')
t_data = np.stack([t], axis=1).astype('float32')
