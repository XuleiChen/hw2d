import jax.numpy as jnp
from jax import jit
from gradients import periodic_gradient
'''
Computing the statistics at time t
(given the grid containing the corresponding value of phi, omega, density at each grid point at t).
Gamma_n: partical flux, measures the rate at which particles enter the system.
Gamma_c: Resistive dissipation, which constitutes the main energy sink.
'''
@jit
def get_gamma_c(n, phi, c1):
    gamma_c=c1*jnp.mean((n-phi)**2,axis=(-1,-2)) # mean over the whole grids
    return gamma_c

@jit
def get_gamma_n(n,phi,dx):
    dy_p=periodic_gradient(phi,dx=dx,axis=-2) # gradient in y
    gamma_n=-jnp.mean((n*dy_p),axis=(-1,-2)) # mean over the whole grids
    return gamma_n
