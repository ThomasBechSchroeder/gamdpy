""" Example of a Simulation using gamdpy, using explicit blocks.

Simulation of a Lennard-Jones crystal in the NVT ensemble followed by shearing with SLLOD 
and Lees-Edwards boundary conditions. Runs one shear rate but easy to make a loop over shear rates.

"""
import os
import numpy as np
import gamdpy as gp
import matplotlib.pyplot as plt

run_NVT = True # False # 

# Setup pair potential: Single component 12-6 Lennard-Jones
pairfunc = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = gp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

temperature_low = 0.700
gridsync = True


if run_NVT:
    # Setup configuration: FCC Lattice
    configuration = gp.Configuration(D=3, compute_flags={'stresses':True})
    configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=2.0)


    # Setup integrator to melt the crystal
    dt = 0.005
    num_blocks = 10
    steps_per_block = 2048
    running_time = dt*num_blocks*steps_per_block

    Ttarget_function = gp.make_function_ramp(value0=2.000, x0=running_time*(1/8),
                                             value1=temperature_low, x1=running_time*(7/8))
    integrator_NVT = gp.integrators.NVT(Ttarget_function, tau=0.2, dt=dt)

    # Setup runtime actions, i.e. actions performed during simulation of timeblocks
    runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]



    # Set simulation up. Total number of timesteps: num_blocks * steps_per_block
    sim_NVT = gp.Simulation(configuration, pairpot, integrator_NVT, runtime_actions,
                            num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                            storage='memory')



    for block in sim_NVT.run_timeblocks():
        print(block)
        print(sim_NVT.status(per_particle=True))

    # save both in hdf5 and rumd-3 formats
    gp.configuration_to_hdf5(configuration, 'LJ_cooled_0.70.h5')

else:
    configuration = gp.configuration_from_hdf5('LJ_cooled_0.70.h5', compute_flags={'stresses':True})

compute_plan = gp.get_default_compute_plan(configuration)
compute_plan['gridsync'] = gridsync
print('compute_plan')
print(compute_plan)
print("Now run SLLOD simulation on what should now be a glass or polycrystal")

sc_output = 8


dt = 0.01
sr = 0.02 # restuls for different values shown in comments below. This value only takes 4 seconds to run so good for running as a test

configuration.simbox = gp.LeesEdwards(configuration.D, configuration.simbox.get_lengths())

integrator_SLLOD = gp.integrators.SLLOD(shear_rate=sr, dt=dt)

# set the kinetic temperature to the exact value associated with the desired
# temperature since SLLOD uses an isokinetic thermostat
configuration.set_kinetic_temperature(temperature_low, ndofs=configuration.N*3-4) # remove one DOF due to constraint on total KE

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
totalStrain = 10.0
steps_per_block = 4096
total_steps = int(totalStrain / (sr*dt)) + 1
num_blocks = total_steps // steps_per_block + 1
strain_transient = 1.0 # how much of the output to ignore
num_steps_transient = int(strain_transient / (sr*dt) ) + 1


# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(include_simbox=True),
                   gp.MomentumReset(100),
                   gp.StressSaver(sc_output, compute_flags={'stresses':True}),
                   gp.ScalarSaver(sc_output)]


print(f'num_blocks={num_blocks}')
sim_SLLOD = gp.Simulation(configuration, pairpot, integrator_SLLOD, runtime_actions,
                          num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                          storage='memory', compute_plan=compute_plan)

# Run simulation one block at a time
for block in sim_SLLOD.run_timeblocks():
    print(sim_SLLOD.status(per_particle=True))
    configuration.simbox.copy_to_host()
    box_shift = configuration.simbox.box_shift
    lengths = configuration.simbox.get_lengths()
    print(f'box-shift={box_shift:.4f}, strain = {box_shift/lengths[1]:.4f}')
print(sim_SLLOD.summary())


U, K, W, V_sxy = gp.extract_scalars(sim_SLLOD.output, ['U', 'K', 'W', 'Sxy'])
N = configuration.N
u, k, sxy = U/N,K/N, V_sxy / configuration.get_volume()

# alternative (newer way) to get the shear stress
full_stress_tensor = gp.extract_stress_tensor(sim_SLLOD.output)
sxy_alt = full_stress_tensor[:,0,1]

times = np.arange(len(u)) * sc_output *  dt
stacked_output = np.column_stack((times, u, k, sxy, sxy_alt))
np.savetxt('shear_run.txt', stacked_output, delimiter=' ', fmt='%f')



strains = times * sr

num_items_transient = num_steps_transient // sc_output
print(f'num_items_transient={num_items_transient}')
sxy_SS = sxy[num_items_transient:]

sxy_mean = np.mean(sxy_SS)
print(f'{sr:.2g} {sxy_mean:.6f}')

#plt.figure(1)
#plt.plot(strains, k)
#plt.plot(time, u)
#plt.figure(2)
#plt.plot(strains, sxy)
#plt.show()

# STRAINRATE VS MEAN STRESS


# 0.001 0.014687
# 0.0025 0.037265
# 0.005 0.071615
# 0.0075 0.111890
# 0.01 0.140723
# 0.02 0.264056
# 0.03 0.363014


# quadratic fit gives the following
# -0.00083871 + 15.436 * x - 110.19*x^2

# The small value of the stress at zero is consistent with zero
# which is promising and we can read the Newtonian viscosity off as 15.436
