""" Example of a simulation of a Lennard-Jones chain with 10 beads per chain """

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gamdpy as gp

# Generate configuration with a FCC lattice
rho = 1.0
configuration = gp.Configuration(D=3, compute_flags={'Fsq':True, 'lapU':True, 'Vol':True})
configuration.make_positions(N=2000, rho=rho)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=1.44)

# Make bonds
bond_potential = gp.harmonic_bond_function
bond_params = [[1.00, 3000.], ]
bond_indices = [[i, i + 1, 0] for i in range(0, configuration.N - 1, 1) if
                i % 10 != 9]  # 10 bead chains (last molecule: N % 10)
bonds = gp.Bonds(bond_potential, bond_params, bond_indices)

# Make pair potential
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
exclusions = bonds.get_exclusions(configuration)
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Define molecules (to be implemented)
molecules_A = [np.arange(0, 10) + j for j in range(0, configuration.N // 2, 10)]
molecules_B = [np.arange(0, 10) + j for j in range(configuration.N // 2, configuration.N, 10)]
# molecules = Molecules([molecules_A, molecules_B], names=['H2O', 'Hexaflourobenzene'])
# molecules.check_identical() # Maybe make check in __init__ and print warning if not all molecules of a type are identical
# molecules.get_numbe_particles()
# molecules.get_resulting_force() # Requires atomic forces to be calculated
#print(molecules_A[:3])


# Make integrator
dt = 0.002  # timestep
num_blocks = 16  # Do simulation in this many 'blocks'
steps_per_block = 2048  # ... each of this many steps
running_time = dt * num_blocks * steps_per_block
temperature = 0.7
filename = 'Data/LJchain10_Rho1.00_T0.700.h5'
Ttarget_function = gp.make_function_ramp(value0=10.000, x0=running_time * (1 / 8),
                                         value1=temperature, x1=running_time * (1 / 4))
integrator0 = gp.integrators.NVT(Ttarget_function, tau=0.2, dt=dt)

compute_plan = gp.get_default_compute_plan(configuration)
print(compute_plan)

runtime_actions = [gp.MomentumReset(100),
                   gp.ConfigurationSaver(),
                   gp.ScalarSaver(32, {'Fsq':True, 'lapU':True}), ]


sim = gp.Simulation(configuration, [pair_pot, bonds], integrator0, runtime_actions,
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    compute_plan=compute_plan, storage=filename)

print('High Temperature followed by cooling and equilibration:')
for block in sim.run_timeblocks():
    if block % 10 == 0:
        print(f'{block=:4}  {sim.status(per_particle=True)}')
print(sim.summary())

integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)
sim = gp.Simulation(configuration, [pair_pot, bonds], integrator, runtime_actions,
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    compute_plan=compute_plan, storage=filename)

# Setup on-the-fly calculation of Radial Distribution Function
calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=1000)

print('Production:')
for block in sim.run_timeblocks():
    if block % 10 == 0:
        print(f'{block=:4}  {sim.status(per_particle=True)}')
    calc_rdf.update()
print(sim.summary())

columns = ['U', 'W', 'K', 'lapU', 'Fsq', 'Vol']
with h5py.File(filename, "r") as f:
    data = np.array(gp.extract_scalars(f, columns, first_block=1))
df = pd.DataFrame(data.T, columns=columns)
df['t'] = np.arange(len(df['U'])) * dt * sim.output['scalar_saver'].attrs["steps_between_output"]
gp.plot_scalars(df, configuration.N, configuration.D, figsize=(10, 8), block=False)

mu = np.mean(df['U']) / configuration.N
mw = np.mean(df['W']) / configuration.N
cvex = np.var(df['U']) / temperature ** 2 / configuration.N

print('gamdpy:')
print(f'Potential energy:     {mu:.4f}')
print(f'Excess heat capacity: {cvex:.3f}')
print(f'Virial                {mw:.4f}')

with h5py.File(filename, "r") as f:
    dynamics = gp.tools.calc_dynamics(f, first_block=0, qvalues=[7.118])
fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.subplots_adjust(hspace=0.00)  # Remove vertical space between axes
axs[0].set_ylabel('MSD')
axs[1].set_ylabel('Non Gaussian parameter')
axs[2].set_ylabel('Intermediate scattering function')
axs[2].set_xlabel('Time')
axs[0].grid(linestyle='--', alpha=0.5)
axs[1].grid(linestyle='--', alpha=0.5)
axs[2].grid(linestyle='--', alpha=0.5)

axs[0].loglog(dynamics['times'], dynamics['msd'], 'o--')
axs[1].semilogx(dynamics['times'], dynamics['alpha2'], 'o--')
axs[2].semilogx(dynamics['times'], dynamics['Fs'], 'o--')
if __name__ == "__main__":
    plt.show(block=False)

rdf = calc_rdf.read()
rdf['rdf'] = np.mean(rdf['rdf'], axis=0)
fig, axs = plt.subplots(1, 1, figsize=(8, 4))
axs.set_ylabel('RDF')
axs.set_xlabel('Distance')
axs.grid(linestyle='--', alpha=0.5)
axs.plot(rdf['distances'], rdf['rdf'], '-')
axs.set_xlim((0.5, 3.5))
if __name__ == "__main__":
    plt.show(block=True)
