""" Example of a Simulation using gamdpy, using explicit blocks.

Simulation of a Lennard-Jones crystal in the NVT ensemble followed by shearing with SLLOD 
and Lees-Edwards boundary conditions. Runs one shear rate but easy to make a loop over shear rates.

"""
import os
import math
import numpy as np
import gamdpy as gp
import matplotlib.pyplot as plt

# set to True only if need to re-make the reference configuration for some reason
run_NVT =  False # True # 

# Setup pair potential: Single component 12-6 Lennard-Jones
pairfunc = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
#pairfunc = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = gp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)


# rho and T taken from Ferrario et al. Phys Rev A, 44, 6936 (1991). Supposed to be LJ triple point
density = 0.8442
start_temperature = 3.0
target_temperature = 0.725
gridsync = True

direc = "../tests/reference_data"
conf_filename = direc + "/" + f'conf_LJ_N2048_rho{density:.4f}_{target_temperature:.4f}.h5'

if run_NVT:
    # Setup configuration: FCC Lattice
    configuration = gp.Configuration(D=3, compute_flags={'stresses':True})
    configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=density)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=start_temperature)


    # Setup integrator to melt the crystal
    dt = 0.005
    num_blocks = 20
    steps_per_block = 4096
    running_time = dt*num_blocks*steps_per_block

    Ttarget_function = gp.make_function_ramp(value0=start_temperature, x0=running_time*(1/8),
                                             value1=target_temperature, x1=running_time*(7/8))
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
    gp.configuration_to_hdf5(configuration, conf_filename)

else:
    configuration = gp.configuration_from_hdf5(conf_filename, compute_flags={'stresses':True})

compute_plan = gp.get_default_compute_plan(configuration)
compute_plan['gridsync'] = gridsync

sc_output = 16
dt = 0.005
sr = 0.16


configuration.simbox = gp.LeesEdwards(configuration.D, configuration.simbox.get_lengths())

integrator_SLLOD = gp.integrators.SLLOD(shear_rate=sr, dt=dt)

# set the kinetic temperature to the exact value associated with the desired
# temperature since SLLOD uses an isokinetic thermostat
configuration.set_kinetic_temperature(target_temperature, ndofs=configuration.N*3-4) # remove one DOF due to constraint on total KE

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
totalStrain = 100.0
steps_per_block = 4096
total_steps = int(totalStrain / (sr*dt)) + 1
num_blocks = total_steps // steps_per_block + 1
strain_transient = 1.0 # how much of the output to ignore (ie 1 corresponds to the first 100% of strain)
num_steps_transient = int(strain_transient / (sr*dt) ) + 1


# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(include_simbox=True),
                   gp.MomentumReset(100),
                   gp.StressSaver(sc_output, compute_flags={'stresses':True}),
                   gp.ScalarSaver(sc_output)]


calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=1000)

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
    calc_rdf.update()
    print(f'box-shift={box_shift:.4f}, strain = {box_shift/lengths[1]:.4f}')
print(sim_SLLOD.summary())




full_stress_tensor = gp.StressSaver.extract(sim_SLLOD.output)
sxy = full_stress_tensor[:,0,1]

times = np.arange(len(full_stress_tensor)) * sc_output *  dt
strains = times * sr
stacked_output = np.column_stack((times, sxy))
np.savetxt('shear_run.txt', stacked_output, delimiter=' ', fmt='%f')


num_items_transient = num_steps_transient // sc_output
print(f'num_items_transient={num_items_transient}')
sxy_SS = sxy[num_items_transient:]

sxy_mean = np.mean(sxy_SS)
sxy_var = np.var(sxy_SS, ddof=1)

time_SS = (totalStrain - strain_transient) / sr
t_corr = 0.3 # estimated visually (no fitting) from calculating autocorrelation of stress in xmgrace using data for SR 0.16.

num_independent =  time_SS / t_corr
error_on_mean_sts = math.sqrt(sxy_var/num_independent)
viscosity = sxy_mean / sr
error_on_visc = error_on_mean_sts/sr
print('SR mean-stress viscosity; errors as two-sigma ie 95% confidence')
print(f'{sr:.2g} {sxy_mean:.6f}+/- {2*error_on_mean_sts:.3f} {viscosity: .4f}+/-{2*error_on_visc:.4}')

# TO DO
# 1. Error bar for one run DONE
# 2. Loop over SR values
# 3. Number and choice of SR values to get convincing check in reasonable time
# 4. Include table of literature values with errors
# 5. Generate graph comparing.
# 6. Figure out issue with committing reference_data 

#calc_rdf.save_average()

data_Ferrario_1991 = np.array([[0.0, 3.24, 0.04],
                               [0.01, 3.20, 0.12],
                               [0.05, 3.16, 0.05],
                               [0.10, 3.13, 0.05],
                               [0.15, 3.01, 0.04],
                               [0.20, 2.88, 0.04],
                               [0.25, 2.81, 0.03],
                               [0.316, 2.67, 0.03],
                               [0.6, 2.37, 0.02],
                               [0.8, 2.22, 0.02],
                               [1.0, 2.12, 0.01],
                               [1.41, 1.95, 0.01],
                               [1.73, 1.84, 0.01],
                               [2.0, 1.76, 0.01]])
plt.errorbar(data_Ferrario_1991[:,0],data_Ferrario_1991[:,1],data_Ferrario_1991[:,2])
plt.show()


# STRAINRATE VS MEAN STRESS
# SR STS VIS [Run 2]
# THESE ARE WITH SF
# 0.01 0.031797 3.1797 [0.031456, 3.1456]
# 0.02 0.063627 3.1813 [0.063490, 3.1745]
# 0.04 0.125902 3.1475 [0.126158, 3.1539]
# 0.08 0.247592 3.0949 [0.248974, 3.1122]
# 0.16 0.469106 2.9318 [0.468686, 2.9293]
# 0.25 0.692740 2.7710 [0.695243, 2.7810]

#Quadratic fit to sts vs SR [first column] for Shifted Force -0.00024556 + 3.2343*SR - 1.8524 * SR**2

# THESE ARE WITH SP
# 0.01 0.031947 3.1947 [0.032859 3.2859 0.033162 3.3162]
# 0.02 0.064094 3.2047 [0.064428 3.2214]
# 0.04 0.129757 3.2439 [0.129340 3.2335]
# 0.08 0.251774 3.1472 [0.253050 3.1631]
# 0.16 0.472397 2.9525  [0.477547 2.9847]
# 0.25 0.697170 2.7887 [0.692740, 2.7889]


# for 0.01 it was 2 million steps


# Compare with Ferrario et al Phys. REv. A, 44, 6936 (1991)
# They also used SLLOD to measure viscosity of the LJ fluid at different
# strain rates. Same density, temperature, N, time step. Cutoff slightly
# different: shifted potential with quadratic smoothing between 2.5 and 2.51.
# 10^7 time steps (my longest is SR 0.01 with 2 million). Also different
# thermostat for SLLOD
