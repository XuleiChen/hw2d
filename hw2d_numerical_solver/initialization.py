import jax.numpy as jnp
from jax import random, jit
from typing import Tuple
from functools import partial
import matplotlib.pyplot as plt
from jax import lax

@partial(jit, static_argnums=0)
def get_fft_noise(
        resolution: Tuple[int, int],
        size: float,
        scale: float,
        min_frequency: float = 0,
        max_frequency: float = 0,
        min_wavelength: float = 0,
        max_wavelength: float = 0,
        factor: int = 2,
        key=random.PRNGKey(0)  # JAX uses a PRNG key for random number generation
) -> jnp.ndarray:
    """
    Generate a 2D noise pattern using the FFT with JAX.

    Args:
        resolution (Tuple[int, int]): The dimensions of the generated noise.
        size (float): Physical size.
        scale (float): Scaling factor for the frequencies.
        min_frequency (float, optional): Minimum frequency for filtering. Default is 0.
        max_frequency (float, optional): Maximum frequency for filtering. Default is 0.
        min_wavelength (float, optional): Minimum wavelength for filtering. Default is 0.
        max_wavelength (float, optional): Maximum wavelength for filtering. Default is 0.
        factor (int, optional): Factor used in calculating the frequency components. Default is 2.

    Returns:
        jnp.ndarray: The generated 2D noise pattern.
    """
    # Calculate random complex values
    shape = (1, *resolution, 1)
    rnd_real = random.normal(key, shape).astype(jnp.complex64)
    rnd_imag = 1j * random.normal(key, shape).astype(jnp.complex64)
    rndj = rnd_real + rnd_imag

    # Calculate frequency components
    k = jnp.meshgrid(*[jnp.fft.fftfreq(n) for n in resolution], indexing="ij")
    k = jnp.expand_dims(jnp.stack(k, -1), 0)
    k = k * jnp.array(resolution) / size * scale  # in physical units
    k = jnp.sum(jnp.abs(k) ** factor, axis=-1, keepdims=True)

    # Convert wavelengths to frequencies if provided
    min_freq = jnp.where(max_wavelength != 0, 1 / max_wavelength, 0)
    max_freq = jnp.where(min_wavelength != 0, 1 / min_wavelength, 0)

    # Create frequency mask
    weight_mask = jnp.ones(shape)
    if min_frequency:
        weight_mask += 1 / (1 + jnp.exp((min_frequency - k) * 1e3)) - 1
    if max_frequency:
        weight_mask -= 1 / (1 + jnp.exp((max_frequency - k) * 1e3))

    # # Check weight mask
    # if jnp.any(weight_mask > 1) or jnp.any(weight_mask < 0):
    #     raise ValueError("Weight mask values out of bounds.")

    # Handle division by zero for k
    k = jnp.where(k == 0, jnp.inf, k)
    inv_k = 1 / k
    inv_k = jnp.where(k == jnp.inf, 0, inv_k)  # Set back to 0 where k was originally 0

    # Compute result
    smoothness = 1
    fft = rndj * inv_k ** smoothness * weight_mask
    array = jnp.real(jnp.fft.ifft2(fft, axes=[1, 2]))
    array /= jnp.std(array, axis=tuple(range(1, len(array.shape))), keepdims=True)
    array -= jnp.mean(array, axis=tuple(range(1, len(array.shape))), keepdims=True)
    array = array.astype(jnp.float32)

    return array[0, ..., 0]

#
# # Example usage
# resolution = (256, 256)
# size = 10.0
# scale = 1.0
# noise_pattern = get_fft_noise(resolution, size, scale)
# plt.figure(figsize=(8, 8))
# plt.imshow(noise_pattern, cmap='gray', origin='lower')
# plt.colorbar(label='Noise Intensity')
# plt.title('Generated 2D FFT Noise Pattern')
# plt.axis('off')  # Hide the axis
# plt.show()
