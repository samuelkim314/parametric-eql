import numpy as np
from generate_data.parametric_pde_find import *
from scipy.integrate import odeint
from numpy.fft import fft, ifft, fftfreq

def ks_rhs(u, t, params):
    k, L, x = params
    
    a = (1 + 0.25*np.sin(x*2*np.pi/L))  # uu_x
    b = -1 + 0.25*np.exp(-(x-2)**2/5)   # u_{xx}
    c = -1 - 0.25*np.exp(-(x+2)**2/5)   # u_{xxxx}
    
    deriv = a*u*ifft(1j*k*fft(u)) + b*ifft(-k**2*fft(u)) + c*ifft(k**4*fft(u))

    return np.real(deriv)

# Set size of grid
n = 256
m = 512
L = 10

# Set up grid
x = np.linspace(-L,L,n+1)[:-1];   dx = x[1]-x[0]
t = np.linspace(0,200,2*m);         dt = t[1]-t[0]
k = 2*np.pi*fftfreq(n, d = dx)

# Initial condition
u0 = np.exp(-(x)**2)

# Solve with time dependent uu_x term
params = (k,L,x)
u_full = odeint(ks_rhs, u0, t, args=(params,)).T
u = u_full[:,-m:]

Ut, Theta, rhs_des = build_linear_system(u, dt, dx, D=4, P=3, time_diff = 'FD', space_diff = 'Fourier')

Theta_grouped = [np.real(Theta[n*np.arange(m)+j,:]) for j in range(n)]
Ut_grouped = [np.real(Ut[n*np.arange(m)+j]) for j in range(n)]

u = np.stack([Tg[:,1] for Tg in Theta_grouped], axis=0)
u_x = np.stack([Tg[:,4] for Tg in Theta_grouped], axis=0)
u_xx = np.stack([Tg[:,8] for Tg in Theta_grouped], axis=0)
u_xxxx = np.stack([Tg[:,16] for Tg in Theta_grouped], axis=0)

u_t = np.stack([Ug[:,0] for Ug in Ut_grouped], axis=0)

in_data = np.stack([u, u_x, u_xx, u_xxxx], axis=2).astype('float32')
out_data = np.stack([u_t], axis=2).astype('float32')
t_data = np.stack([x], axis=1).astype('float32')