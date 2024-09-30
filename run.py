from model import HW
from initialization import get_fft_noise
import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial
import matplotlib.pyplot as plt
import os

step_size: float = 0.02
end_time: float = 70
initial_time: float = 0
grid_pts: int = 1024
k0: float = 0.15
N: int = 2
nu: float = 5.0e-08
c1: float = 1.0
kappa: float = 1.0

y = grid_pts
x = grid_pts
L = 2 * jnp.pi / k0
dx = L / x
steps = int((end_time - initial_time) / step_size)

# Fourier noise as initial state，initialize phi, omega, density from noise
# Set up initial values
hw=HW(dx=dx,N=N,nu=nu,c1=c1,kappa=kappa)
initial_omega = get_fft_noise(resolution=(x, y), size=L, scale=1, min_wavelength=dx * 1,
                      max_wavelength=dx * grid_pts * 10)
initial_density = get_fft_noise(resolution=(x, y), size=L, scale=1, min_wavelength=dx * 1,
                        max_wavelength=dx * grid_pts * 10)
initial_phi=hw.get_phi(omega=initial_omega,dx=dx)

@partial(jit, static_argnums=(3,))
def run(
    omega,density,phi,steps
):
    # Run simulation
    def run_iterations(carry, i):
        density, omega, phi = carry
        # 进行一次rk4_step更新
        density, omega, phi = hw.rk4_step(dt=step_size, dx=dx, pn=phi, n=density, o=omega)
        # 返回新的carry和输出值，其中输出值记录每个step的density, omega和phi
        return (density, omega, phi),(density, omega, phi)

    # 使用lax.scan进行迭代
    initial_carry = (density, omega, phi)
    final_carry, outputs = lax.scan(run_iterations, initial_carry, jnp.arange(steps))

    # 提取最后的记录
    density_values, omega_values, phi_values =outputs

    return phi_values, density_values, omega_values

phi_values, density_values, omega_values=run(
    omega=initial_omega, phi=initial_phi,density=initial_density,steps=steps
)

# Plot the result
# Extracting the last step values
final_phi = phi_values[-1]
final_omega = omega_values[-1]
final_density = density_values[-1]

# Function to plot the final omega, phi, and density
def plot_final_fields(final_omega, final_phi, final_density,save_path=None):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # final omega
    ax[0].imshow(final_omega, cmap='viridis')
    ax[0].set_title("Final Omega (vorticity field)")
    ax[0].axis('off')

    # final phi
    ax[1].imshow(final_phi, cmap='inferno')
    ax[1].set_title("Final Phi (stream function)")
    ax[1].axis('off')

    # final density
    ax[2].imshow(final_density, cmap='plasma')
    ax[2].set_title("Final Density")
    ax[2].axis('off')

    if save_path is not None:
        plt.savefig(save_path)
        plt.show()
        plt.close()
    else:
        plt.show()


# Define the folder name
folder_name = "results"

# Create the folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Save results as .npy files inside the folder
jnp.save(os.path.join(folder_name, 'phi_values.npy'), phi_values)
jnp.save(os.path.join(folder_name, 'density_values.npy'), density_values)
jnp.save(os.path.join(folder_name, 'omega_values.npy'), omega_values)

# Save the plot as a PNG file inside the folder
plot_final_fields(final_omega, final_phi, final_density, save_path=os.path.join(folder_name, "final_fields.png"))