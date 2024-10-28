import os
import time
import h5py
from model import HW
from initialization import get_fft_noise
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
import numpy as np
import jax

# Create results folder if not exists
folder_name = '/scratch/xc2695/hw2d_numerical_solver/hw2d_solver/results'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Simulation parameters
start_time = time.time()
step_size: float = 0.015
end_time: float = 1000
initial_time: float = 0
grid_pts: int = 256
k0: float = 0.15
N: int = 3
nu: float = 5.0e-08
c1: float = 1.0
kappa: float = 1.0
initial_scale = 0.01

y = grid_pts
x = grid_pts
L = 2 * jnp.pi / k0
dx = L / x
steps = int((end_time - initial_time) / step_size)

# Fourier noise as initial state, initialize phi, omega, density from noise
hw = HW(dx=dx, N=N, nu=nu, c1=c1, kappa=kappa)
initial_omega = get_fft_noise(resolution=(x, y), size=L, scale=1, min_wavelength=dx * 12,
                              max_wavelength=dx * grid_pts * 100) * initial_scale
initial_density = get_fft_noise(resolution=(x, y), size=L, scale=1, min_wavelength=dx * 1,
                                max_wavelength=dx * grid_pts * 10) * initial_scale
initial_phi = hw.get_phi(omega=initial_omega, dx=dx)

# Set the .h5 file paths to the results folder
h5_filename_train = os.path.join(folder_name, 'hasegawa_wakatani_simulation_data_train.h5')
h5_filename_validation = os.path.join(folder_name, 'hasegawa_wakatani_simulation_data_validation.h5')
h5_filename_test = os.path.join(folder_name, 'hasegawa_wakatani_simulation_data_test.h5')

# Run simulation and save the data to .h5 file
with h5py.File(h5_filename_train, 'w') as f_train, \
     h5py.File(h5_filename_validation, 'w') as f_validation, \
     h5py.File(h5_filename_test, 'w') as f_test:

    @partial(jit, static_argnums=(3,))
    def run(omega, density, phi, steps):
        def run_iterations(carry, i):
            density, omega, phi = carry
            density, omega, phi, _, _ = hw.rk4_step(dt=step_size, dx=dx, pn=phi, n=density, o=omega)
            return (density, omega, phi), (density, omega, phi)

        initial_carry = (density, omega, phi)
        final_carry, outputs = lax.scan(run_iterations, initial_carry, jnp.arange(steps))
        density_vals, omega_vals, phi_vals = outputs
        final_density, final_omega, final_phi = final_carry
        return density_vals, omega_vals, phi_vals, final_density, final_omega, final_phi

    # Run the simulation for all steps
    density_vals, omega_vals, phi_vals, final_density, final_omega, final_phi = run(
        omega=initial_omega, phi=initial_phi, density=initial_density, steps=steps
    )

    train_end = int(0.8 * len(density_vals))
    validation_end = int(0.9 * len(density_vals))

    f_train.create_dataset('density', data=np.array(density_vals[:train_end]), dtype='float32')
    f_train.create_dataset('omega', data=np.array(omega_vals[:train_end]), dtype='float32')
    f_train.create_dataset('phi', data=np.array(phi_vals[:train_end]), dtype='float32')

    f_validation.create_dataset('density', data=np.array(density_vals[train_end:validation_end]), dtype='float32')
    f_validation.create_dataset('omega', data=np.array(omega_vals[train_end:validation_end]), dtype='float32')
    f_validation.create_dataset('phi', data=np.array(phi_vals[train_end:validation_end]), dtype='float32')

    f_test.create_dataset('density', data=np.array(density_vals[validation_end:]), dtype='float32')
    f_test.create_dataset('omega', data=np.array(omega_vals[validation_end:]), dtype='float32')
    f_test.create_dataset('phi', data=np.array(phi_vals[validation_end:]), dtype='float32')

    print(f"Density dataset shape: {f_train['density'].shape}")
    print(f"Omega dataset shape: {f_validation['omega'].shape}")
    print(f"Phi dataset shape: {f_test['phi'].shape}")



# Print running time
end_time = time.time()
total_runtime = end_time - start_time
print(f"Total runtime: {total_runtime:.2f} seconds")




import matplotlib.pyplot as plt

# Function to plot the final omega, phi, and density
def plot_final_fields(final_omega, final_phi, final_density, save_path=None):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot final omega
    im0 = ax[0].imshow(final_omega, cmap='viridis')
    ax[0].set_title("Final Omega (Vorticity Field)")
    ax[0].axis('off')
    fig.colorbar(im0, ax=ax[0])

    # Plot final phi
    im1 = ax[1].imshow(final_phi, cmap='inferno')
    ax[1].set_title("Final Phi (Stream Function)")
    ax[1].axis('off')
    fig.colorbar(im1, ax=ax[1])

    # Plot final density
    im2 = ax[2].imshow(final_density, cmap='plasma')
    ax[2].set_title("Final Density")
    ax[2].axis('off')
    fig.colorbar(im2, ax=ax[2])

    plt.savefig(os.path.join(folder_name, 'final_fields.png'))
    plt.close()


# Plot the final fields
plot_final_fields(final_omega, final_phi, final_density)