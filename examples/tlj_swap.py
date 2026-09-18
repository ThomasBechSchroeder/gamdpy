""" Example of a ternary LJ simulation using gamdpy, using swap Monte-Carlo moves.

"""

import gamdpy as gp
import numpy as np
import matplotlib.pyplot as plt

# Specify statepoint
num_part = 1002
rho = 1.35
temperature = 0.652

# Setup configuration: 
configuration = gp.Configuration(D=3)
configuration.make_positions(N=num_part, rho=rho)
configuration['m'] = 1.0 # Specify all masses to unity 
configuration.randomize_velocities(temperature=5.0) # Initial high temperature for randomizing
configuration.ptype[0::6] = 1 # Every sixth particle set to type 1 (4:1:1 mixture)
configuration.ptype[1::6] = 2 # Every sixth particle set to type 2 (4:1:1 mixture)

# Setup pair potential: Binary Ternary LJ mixture.
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[1.00, 0.80, 0.90],
       [0.80, 0.88, 0.84],
       [0.90, 0.84, 0.94]]
eps = [[1.00, 1.50, 1.25],
       [1.50, 0.50, 1.00], 
       [1.25, 1.00, 0.75]]
cut = np.array(sig)*2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Increase 'num_blocks' for longer runs, better statistics, AND bigger storage consumption
# Increase 'steps_per_block' for longer runs
dt = 0.004  # timestep
num_timeblocks = 32           # Do simulation in this many timeblocks. 
steps_per_timeblock = 1*1024   # ... each of this many steps
filename = f'Data/TLJ_Rho{rho:.3f}_T{temperature:.3f}_swap.h5'

integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)

# Setup particle swapper to perform particle Swap Monte-Carlo every 10'th MD step, with 2*N swap attempts per MC turn
swap_types = [[False, False, True], 
              [False, False, True], 
              [True,  True,  False]]

swapper = gp.ParticleSwapper(steps_between_swaps=10, swap_attempts=2*configuration.N, allowed_swap_types=swap_types, 
                             pair_potential=pair_pot, swap_temperature=temperature, seed=42)

#Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [swapper,
                   gp.TrajectorySaver(scheduler=gp.Log2()),
                   gp.ScalarSaver(32),
                   gp.RestartSaver(),
                   gp.MomentumReset(100)]

sim = gp.Simulation(configuration, [pair_pot, ], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    storage=filename)

print(sim.compute_plan)

Us = []
describtion = ['Equilibration', 'Production']
for i in range(2):
    print(f'\n{describtion[i]}')
    for block in sim.run_timeblocks():
        print(f'{sim.status(per_particle=True)}')
    print(sim.summary())

    num_swap_attempts = swapper.swap_attempts*num_timeblocks*steps_per_timeblock//swapper.steps_between_swap
    num_successful_swaps = int(swapper.success_counter[0])
    print(f'{num_swap_attempts=}')
    print(f'{num_successful_swaps=}')
    print(f'Swap acceptance rate: {num_successful_swaps/num_swap_attempts:.3e}')
    print(f'Swap attempts per second: {num_swap_attempts/(sim.timing_numba / 1000):.3e}')
    print()
    TPS_swap = num_timeblocks*steps_per_timeblock/(sim.timing_numba / 1000)
    U, = gp.ScalarSaver.extract(sim.output, columns=['U',], per_particle=True, first_block=0)
    Us.append(U)
    times = gp.ScalarSaver.get_times(sim.output, first_block=0)
    dynamics_swap = gp.tools.calc_dynamics(sim.output, 0, qvalues=7.5, overlap_distances=0.3, extra_times_method='auto')

    swapper.reset_success_counter() # Reset the success counter for the next run

# Print current status of configuration
print(configuration)

# Switch to normal MD for comparison
filename = f'Data/TLJ_Rho{rho:.3f}_T{temperature:.3f}.h5'

#Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(scheduler=gp.Log2()),
                   gp.ScalarSaver(32),
                   gp.RestartSaver(),
                   gp.MomentumReset(100)]

sim = gp.Simulation(configuration, [pair_pot, ], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock*8,
                    storage=filename)

print('Production, standard MD')
for block in sim.run_timeblocks():
    print(f'{sim.status(per_particle=True)}')
print(sim.summary())


# Print current status of configuration
print(configuration)
TPS_md = sim.num_blocks*sim.steps_per_block/(sim.timing_numba / 1000)

U_md,= gp.ScalarSaver.extract(sim.output, columns=['U',], per_particle=True, first_block=0, last_block=num_timeblocks//4)
times_md = gp.ScalarSaver.get_times(sim.output, first_block=0, last_block=num_timeblocks//4)
dynamics_md = gp.tools.calc_dynamics(sim.output, 0, qvalues=7.5, overlap_distances=0.3, extra_times_method='auto')

fig, axs = plt.subplots(2, 1, figsize=(8, 9), sharex=False)
axs[0].set_ylabel('p(U/N)')
axs[1].set_ylabel('MSD')
axs[0].set_xlabel('U/N')
axs[1].set_xlabel('Time')
axs[0].grid(linestyle='--', alpha=0.5)
axs[1].grid(linestyle='--', alpha=0.5)

#axs[0].plot(times, Us[0], label='Swap MC ' + describtion[0])
#axs[0].plot(times, Us[1], label='Swap MC ' + describtion[1])
#axs[0].plot(times_md, U_md, label='NVT')
#axs[0].set_ylim( (np.mean(Us[1]) - 5*np.std(Us[1]), np.mean(Us[1]) + 15*np.std(Us[1])))
n_bins = 30
axs[0].hist(Us[1], n_bins, histtype='step', density=True, stacked=True, fill=False, label='Swap MC')
axs[0].hist(U_md, n_bins, histtype='step', density=True, stacked=True, fill=False, label='MD')
axs[0].legend()

factor = np.array([1, 30])
axs[1].loglog(dynamics_swap['times'], dynamics_swap['msd'][:,0], 'o--', label='Swap MC ' + describtion[1] + f' TPS={TPS_swap:.2e}')
axs[1].loglog(dynamics_md['times'], dynamics_md['msd'][:,0], 'o--', label='NVT' + f' TPS={TPS_md:.2e}')

Dswap = dynamics_swap['msd'][-1,0]/dynamics_swap['times'][-1]/6
axs[1].loglog(dynamics_swap['times'][-1]/factor, 6*Dswap*dynamics_swap['times'][-1]/factor, 'k-.', alpha=0.5, label=f'Slope 1 (D={Dswap:.2e}))')

Dmd = dynamics_md['msd'][-1,0]/dynamics_md['times'][-1]/6
axs[1].loglog(dynamics_md['times'][-1]/factor, 6*Dmd*dynamics_md['times'][-1]/factor, 'k-.', alpha=0.5, label=f'Slope 1 (D={Dmd:.2e}))')
axs[1].legend()

plt.show()

