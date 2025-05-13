""" Example of computing the structure factor of a Lennard-Jones liquid.
    S(𝐪) = 1/N * |sum_{i=1}^{N} exp(-i𝐪•𝐫)|^2
    where 𝐪 is the wave vector, 𝐫 is the position of the particles, and N is the number of particles.
    The 𝐪 vectors are given by 𝐪 = 2π𝐧/L, where 𝐧 is a vector of integers and L is the box size.
"""
import matplotlib.pyplot as plt
import numpy as np

import gamdpy as gp

# Setup simulation of single-component Lennard-Jones liquid
temperature: float = 2.0
density: float = 0.973
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=density)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=1.44)
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_potential = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)
integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=0.005)
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]


sim = gp.Simulation(configuration, pair_potential, integrator, runtime_actions,
                    num_timeblocks=16, steps_per_timeblock=512,
                    storage='memory')

print("Equilibration run")
sim.run()

print("Production run")
q_max = 18.0
calc_struct_fact = gp.CalculatorStructureFactor(configuration, backend='GPU')
calc_struct_fact.generate_q_vectors(q_max=q_max)
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    calc_struct_fact.update()
print(sim.summary())

struct_fact = calc_struct_fact.read(bins=100)
q = struct_fact['|q|']
S = struct_fact['S(|q|)']

# Find maximum of S(q) and corresponding q
max_S_raw = np.max(S)
max_q_raw = q[np.argmax(S)]
# Fit polynomial to two nearest points to
n_range = 2
idx_max = np.argmax(S)
idx_fit = slice(idx_max-n_range, idx_max+n_range)
fit = np.polyfit(q[idx_fit], S[idx_fit], 2)
a, b, c = fit
max_q = -b / (2 * a)
max_S = a * max_q ** 2 + b * max_q + c

plt.figure()
plt.title(f'Lennard-Jones liquid at $T={temperature}$ and $\\rho={density}$')
plt.plot(q, S, 'o')
x_fit = np.arange(q[idx_fit.start], q[idx_fit.stop], 0.01)
plt.plot(x_fit, a * x_fit ** 2 + b * x_fit + c, 'r--')
plt.text(max_q, max_S+1, f'Maximum: $S(q = {max_q:.3f}) = {max_S:.2f}$')
# Use S(q) in q→0 limit and estimate compressibility
plt.plot([0, 1], [S[0], S[0]], 'k--')
kappa_T = S[0]/(temperature*density)
plt.text(1, S[0]/2, r'$\kappa_T=\frac{S(q→0)}{\rho k_B T}=$' f'{kappa_T:.4f}')
plt.yscale('log')
plt.xlabel(r'Length of wave-vector, $|q|$ ($\sigma^{-1}$)')
plt.ylabel(r'Static structure factor, $S(|q|)$')
plt.ylim(1e-2, 10)
plt.xlim(0, max(q))
if __name__ == "__main__":
    plt.show()

