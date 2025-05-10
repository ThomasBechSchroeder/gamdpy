""" Investigation of thermodynamic properties

This example show how thermodynamic data can be extracted
using the `extract_scalars` function from the `gamdpy` package.

The script runs a NVT simulation of a Lennard-Jones crystal.
The potential energy, virial, and kinetic energy are extracted
from the simulation output and the mean values are printed.
Derived quantities such as kinetic temperature and pressure are computed.
The energy trajectory is plotted and the error estimate of potential energy
is estimated using the blocking method.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import gamdpy as gp


###############################
#  Setup and run simulation   #
###############################

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = gp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.ConfigurationSaver(),
                   gp.ScalarSaver(4),
                   gp.MomentumReset(100)]

# Setup Simulation.
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=64, steps_per_timeblock=2048,
                    storage='memory')

# Run simulation
sim.run()

# Basic information of NVT simulation
N = sim.configuration.N  # Number of particles
T = sim.integrator.temperature  # Temperature
V = configuration.get_volume()
rho = N / V  # Density


########################
#  Extracting scalars  #
########################

# Extract potential energy (U), virial (W), and kinetic energy (K)
# We use first_block=1 to skip the initial "equilibration" block.
U, W, K = gp.extract_scalars(sim.output, ['U', 'W', 'K'], first_block=1)

# Print mean values
print(f"Mean potential energy per particle: {np.mean(U) / N}")
print(f"Mean virial per particle: {np.mean(W) / N}")
print(f"Mean kinetic energy per particle: {np.mean(K) / N}")


########################
#  Derived quantities  #
########################

# Time
dt = sim.integrator.dt
time = np.arange(len(U)) * dt * sim.output.attrs["scalar_saver/steps_between_output"]
print(f"Total time of analysed trajectory: {time[-1]}")

# Compute kinetic temperature
D = sim.configuration.D  # dimension of space
dof = D * N - D  # degrees of freedom
T_kin = 2 * K / dof
print(f"Mean kinetic temperature: {np.mean(T_kin)}")

# Compute instantaneous pressure
P = rho * T_kin + W / V
print(f"Mean pressure: {np.mean(P)}")


################################
#  Plotting energy trajectory  #
################################

# Plot potential energy per particle
plt.figure()
plt.plot(time, U / N, label='Potential energy')
plt.axhline(np.mean(U) / N, color='k', linestyle='--', label='Mean')
plt.xlabel('Time')
plt.ylabel('Potential energy')
plt.legend()
plt.show()


########################################
# Statistical analysis using blocking  #
########################################

def error_estimate_by_blocking(data, blocks, confidence=0.95):
    """ Estimate error on mean using block averaging. """
    block_length = len(data) // blocks
    data = data[:blocks * block_length].reshape(blocks, block_length)
    block_means = np.mean(data, axis=1)  # Mean of each block
    sem = np.std(block_means) / np.sqrt(blocks)  # Standard error of the mean
    t_value = stats.t.ppf((1 + confidence) / 2, blocks - 1)  # t-value for confidence interval
    return t_value * sem  # Error estimate


# Find optimal block size
test_num_blocks = list(set(np.logspace(0, np.log10(len(U) / 4), num=32, dtype=int)))
test_num_blocks.sort()
test_num_blocks = test_num_blocks[2:]
errors = [error_estimate_by_blocking(U/N, blocks) for blocks in test_num_blocks]

# Good choice for number of blocks (see figure below)
#   If num_blocks is too small, there is large uncertainty on the error estimate.
#   If num_blocks is too large, blocks are not statistical independent.
num_blocks = 64
error = error_estimate_by_blocking(U/N, num_blocks)

plt.figure()
plt.title('Error estimate of potential energy per particle')
plt.plot(test_num_blocks, errors, 'o')
plt.plot(num_blocks, error, 'ro', label='Estimated error')
plt.xscale('log')
plt.xlim(1, None)
plt.ylim(0, None)
plt.xlabel('Number of blocks')
plt.ylabel('Estimated error (95% confidence interval)')
plt.show()

print(f'Potential energy per particle {np.mean(U) / N:.4f} ± {error:.4f} (95% confidence interval)')
