import numba
import numpy as np
import gamdpy as gp
import matplotlib.pyplot as plt

rho = 0.6
Tinitial = 2.5
Tfinal = 2.0

num_timeblocks = 128              # Do simulation in this many 'timeblocks'
steps_per_timeblock = 2*1024      # ... each of this many steps

# Generate configuration with a FCC lattice
configuration = gp.Configuration(D=3, compute_flags={'Fsq':True, 'lapU':True, 'Vol':True})
configuration.make_lattice(gp.unit_cells.FCC, cells=[6, 6, 6], rho=rho)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=Tinitial)

# Make pair potential
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Make integrator
dt = 0.005  # timestep

running_time = dt*num_timeblocks*steps_per_timeblock
temperature = gp.make_function_ramp(value0=Tinitial, x0=0.25*running_time, value1=Tfinal, x1=0.75*running_time)
integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)

runtime_actions = [gp.MomentumReset(100),
                   gp.TrajectorySaver(),
                   gp.ScalarSaver(32, {'Fsq':True, 'lapU':True}), ]


sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    storage='memory')

# Setup on-the-fly calculation of Radial Distribution Function
calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=300)

dump_filename = 'Data/LJphases_dump.lammps'
with open(dump_filename, 'w') as f:
    print(gp.configuration_to_lammps(sim.configuration, timestep=0), file=f)

for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    with open(dump_filename, 'a') as f:
        print(gp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

    if block > sim.num_blocks*3/4:
        calc_rdf.update()
print(sim.summary())


# Plot data
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=False)
fig.subplots_adjust(hspace=0.20)
axs[0].set_title(f'LJ, {rho=:.3}, {Tfinal=:.3}')

U, W, K = np.array(gp.extract_scalars(sim.output, ['U', 'W', 'K'], first_block=1))
time = np.arange(len(U))* dt * sim.output.attrs["steps_between_output"]
if callable(temperature):
    Ttarget = numba.vectorize(temperature)(time)
    axs[0].grid(linestyle='--', alpha=0.5)
    axs[0].plot(Ttarget, U/configuration.N)
    axs[0].set_xlabel('Temperature')
    axs[0].set_ylabel('Potenital energy per particle')

rdf = calc_rdf.read()
axs[1].grid(linestyle='--', alpha=0.5)
axs[1].plot(rdf['distances'], np.mean(rdf['rdf'], axis=0))
axs[1].set_xlabel('Distance')
axs[1].set_ylabel('Radial distribution function')

dynamics = gp.tools.calc_dynamics(sim.output, first_block=int(sim.num_blocks*3/4), qvalues=7.5*rho**(1/3))
axs[2].grid(linestyle='--', alpha=0.5)
axs[2].loglog(dynamics['times'], dynamics['msd'], 'o-')
factor = np.array([1, 100])
axs[2].loglog(dynamics['times'][0]*factor, np.max(dynamics['msd'][0,:])*factor**2, 'k--', alpha=0.5, label='Slope 2')
axs[2].loglog(dynamics['times'][-1]/factor, np.min(dynamics['msd'][-1,:])/factor, 'k-.', alpha=0.5, label='Slope 1')
axs[2].legend()
axs[2].set_xlabel('Time')
axs[2].set_ylabel('MSD')
plt.savefig('Data/LJphases.pdf')
print('Wrote Data/LJphases.pdf')
#plt.show()

print('\nVisualize simulation in ovito with:')
print('   ovito Data/LJphases_dump.lammps')
