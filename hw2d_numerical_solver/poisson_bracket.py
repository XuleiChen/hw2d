import jax.numpy as jnp
from jax import jit
@jit
def arakawa_vec(zeta, psi, dx):
    """
    Compute the Poisson bracket (Jacobian) of vorticity and stream function
    using a vectorized version of the Arakawa scheme in JAX. This function
    is designed for a 2D periodic domain and requires a 1-cell padded input
    on each border.

    Args:
        zeta (jnp.ndarray): Vorticity field without padding.(N,N)
        psi (jnp.ndarray): Stream function field without padding.(N,N)
        dx (float): Grid spacing.

    Returns:
        jnp.ndarray: Discretized Poisson bracket (Jacobian) over the grid.
    """
    # pad zeta and psi
    zeta = jnp.pad(zeta, pad_width=1, mode='wrap')
    psi = jnp.pad(psi, pad_width=1, mode='wrap')
    return (
        zeta[1:-1, 2:] * (psi[2:, 1:-1] - psi[0:-2, 1:-1] + psi[2:, 2:] - psi[0:-2, 2:])
        - zeta[1:-1, 0:-2] * (psi[2:, 1:-1] - psi[0:-2, 1:-1] + psi[2:, 0:-2] - psi[0:-2, 0:-2])
        - zeta[2:, 1:-1] * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[2:, 2:] - psi[2:, 0:-2])
        + zeta[0:-2, 1:-1] * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[0:-2, 2:] - psi[0:-2, 0:-2])
        + zeta[0:-2, 2:] * (psi[1:-1, 2:] - psi[0:-2, 1:-1])
        + zeta[2:, 2:] * (psi[2:, 1:-1] - psi[1:-1, 2:])
        - zeta[2:, 0:-2] * (psi[2:, 1:-1] - psi[1:-1, 0:-2])
        - zeta[0:-2, 0:-2] * (psi[1:-1, 0:-2] - psi[0:-2, 1:-1])
    ) / (12 * dx**2)
