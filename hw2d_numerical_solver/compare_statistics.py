import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
from statistics import get_gamma_c, get_gamma_n

def compute_statistics(file_density, file_omega, dx, c1):
    with h5py.File(file_density, 'r') as f_density, h5py.File(file_omega, 'r') as f_omega:
        n = jnp.array(f_density['predictions'])
        phi = jnp.array(f_omega['predictions'])
    assert n.shape == phi.shape, "Shape mismatch between n and phi"

    gamma_n_results = []
    gamma_c_results = []

    for batch_idx in range(n.shape[0]):  # 333
        for t in range(n.shape[-1]):  # 19
            n_t = n[batch_idx, ..., t]
            phi_t = phi[batch_idx, ..., t]

            gamma_c_t = get_gamma_c(n_t, phi_t, c1)
            gamma_n_t = get_gamma_n(n_t, phi_t, dx)

            gamma_c_results.append(gamma_c_t)
            gamma_n_results.append(gamma_n_t)

    gamma_c_results = jnp.array(gamma_c_results)
    gamma_n_results = jnp.array(gamma_n_results)

    return gamma_c_results, gamma_n_results

def compute_statistics_hasegawa(file_data, dx, c1):
    with h5py.File(file_data, 'r') as f:
        n = jnp.array(f['density'])
        phi = jnp.array(f['phi'])

    assert n.shape == phi.shape, "Shape mismatch between n and phi"

    gamma_c_results = []
    gamma_n_results = []

    for t in range(n.shape[0]):
        n_t = n[t]
        phi_t = phi[t]

        gamma_c_t = get_gamma_c(n_t, phi_t, c1)
        gamma_n_t = get_gamma_n(n_t, phi_t, dx)

        gamma_c_results.append(gamma_c_t)
        gamma_n_results.append(gamma_n_t)

    return jnp.array(gamma_c_results), jnp.array(gamma_n_results)
dx = 2 * jnp.pi / (0.15 * 256)
c1 = 1.0

file_density = '../hw2d_fourierflow_predictions_density.h5'
file_phi = '../hw2d_fourierflow_predictions_phi.h5'
gamma_c_ff, gamma_n_ff = compute_statistics(file_density, file_phi, dx, c1)

file_data_hasegawa = '../hasegawa_wakatani_simulation_data_test.h5'
gamma_c_hw, gamma_n_hw = compute_statistics_hasegawa(file_data_hasegawa, dx, c1)

gamma_c_ff_mean = jnp.mean(gamma_c_ff)
gamma_c_ff_std = jnp.std(gamma_c_ff)
gamma_n_ff_mean = jnp.mean(gamma_n_ff)
gamma_n_ff_std = jnp.std(gamma_n_ff)

gamma_c_hw_mean = jnp.mean(gamma_c_hw)
gamma_c_hw_std = jnp.std(gamma_c_hw)
gamma_n_hw_mean = jnp.mean(gamma_n_hw)
gamma_n_hw_std = jnp.std(gamma_n_hw)

plt.figure(figsize=(12, 6))
plt.plot(gamma_c_ff, label='Gamma_c (FourierFlow)', color='blue')
plt.plot(gamma_c_hw, label='Gamma_c (Numerical Solve)', color='orange')
plt.title('Gamma_c')
plt.xlabel('Time Steps')
plt.ylabel('Gamma_c Values')
plt.legend(title=f'FourierFlow: {gamma_c_ff_mean:.2f} ± {gamma_c_ff_std:.2f}\nHasegawa: {gamma_c_hw_mean:.2f} ± {gamma_c_hw_std:.2f}')
plt.grid()
plt.savefig('results/gamma_c_comparison.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(gamma_n_ff, label='Gamma_n (FourierFlow)', color='blue')
plt.plot(gamma_n_hw, label='Gamma_n (Numerical Solver)', color='orange')
plt.title('Gamma_n')
plt.xlabel('Time Steps')
plt.ylabel('Gamma_n Values')
plt.legend(title=f'FourierFlow: {gamma_n_ff_mean:.2f} ± {gamma_n_ff_std:.2f}\nHasegawa: {gamma_n_hw_mean:.2f} ± {gamma_n_hw_std:.2f}')
plt.grid()
plt.savefig('results/gamma_n_comparison.png')
plt.close()