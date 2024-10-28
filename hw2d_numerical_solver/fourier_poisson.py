from jax import jit
import jax.numpy as jnp
import jax


@jit
def fourier_poisson_single(tensor: jnp.ndarray, dx: float, times: int = 1) -> jnp.ndarray:
    """Inverse operation to `fourier_laplace`."""
    tensor = jax.device_put(jnp.array(tensor, dtype=jnp.complex64))
    frequencies = jnp.fft.fft2(tensor)

    k = jnp.meshgrid(*[jnp.fft.fftfreq(int(n)) for n in tensor.shape], indexing="ij")
    k = jnp.stack(k, -1)
    k = jnp.sum(k ** 2, axis=-1)
    fft_laplace = -((2 * jnp.pi) ** 2) * k

    result = frequencies / (fft_laplace ** times)
    result = jnp.where(fft_laplace == 0, 0, result)
    result = jnp.real(jnp.fft.ifft2(result))

    return (result * dx**2).astype(jnp.float32)
