import jax.numpy as jnp
from jax import jit
from functools import partial
from jax import lax

# Compute laplace
@jit
def laplace(padded: jnp.ndarray, dx: float):
    """
    Compute the Laplace of a 2D array using finite differences with JAX.
    """
    return (
        padded[...,0:-2, 1:-1]  # above
        + padded[...,1:-1, 0:-2]  # left
        - 4 * padded[...,1:-1, 1:-1]  # center
        + padded[...,1:-1, 2:]  # right
        + padded[...,2:, 1:-1]  # below
    ) / dx**2

@jit
def periodic_laplace(arr: jnp.ndarray, dx: float):
    """
    Compute the Laplace of a 2D array using finite differences with periodic boundary conditions.
    """
    pad_size = (1, 1)
    padded_arr = jnp.pad(arr, pad_width=((1, 1), (1, 1)), mode='wrap')
    return laplace(padded_arr, dx)

@partial(jit, static_argnums=(2,))
def periodic_laplace_N(arr: jnp.ndarray, dx: float, N: int) -> jnp.ndarray:
    """
    Compute the Laplace of a 2D array using finite differences N times successively with periodic boundary conditions.
    """
    for _ in range(N):
        arr = periodic_laplace(arr, dx)
    return arr

# Compute gradient
@jit
def gradient(padded: jnp.ndarray, dx: float, axis: int = 0) -> jnp.ndarray:
    """
        Compute the gradient of a 2D array using central finite differences.
    """
    # if axis == 0:
    return (padded[2:, 1:-1] - padded[0:-2, 1:-1]) / (2 * dx)
    # elif axis == 1:
    #     return (padded[1:-1, 2:] - padded[1:-1, 0:-2]) / (2 * dx)
    # elif axis == -2:
    #     return (padded[..., 2:, 1:-1] - padded[..., 0:-2, 1:-1]) / (2 * dx)
    # elif axis == -1:
    #     return (padded[..., 1:-1, 2:] - padded[..., 1:-1, 0:-2]) / (2 * dx)

@jit
def padsize_generation(input_field: jnp.ndarray,axis:int=0) -> jnp.ndarray:
    pad_size = lax.cond(
        axis < 0,
        lambda: [(0, 0) for _ in range(len(input_field.shape)-2)] + [(1, 1), (1, 1)],
        lambda: [(1,1) for _ in range(len(input_field.shape))]
    )
    return pad_size

@partial(jit, static_argnums=(1,))
def periodic_gradient(input_field: jnp.ndarray,dx: float, axis: int = 0) -> jnp.ndarray:
    """
    Compute the gradient of a 2D array using finite differences with periodic boundary conditions.
    """
    # if axis < 0:
    #     pad_size = [(0, 0) for _ in range(len(input_field.shape))]
    #     pad_size[-1] = (1, 1)
    #     pad_size[-2] = (1, 1)
    # else:
    #     pad_size = 1

    padded = jnp.pad(input_field, pad_width=1, mode="wrap")
    return gradient(padded, dx, axis=axis)