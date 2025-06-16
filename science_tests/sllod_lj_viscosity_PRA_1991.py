""" Example of a Simulation using gamdpy, using explicit blocks.

Simulation of a Lennard-Jones crystal in the NVT ensemble followed by shearing with SLLOD 
and Lees-Edwards boundary conditions. Runs one shear rate but easy to make a loop over shear rates.

"""
import os
import math
import numpy as np
import gamdpy as gp
import matplotlib.pyplot as plt
import scipy.stats

# set to True only if need to re-make the reference configuration for some reason
run_NVT =  False # True # 

# Setup pair potential: Single component 12-6 Lennard-Jones
pairfunc = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
#pairfunc = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = gp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

filename = 'sllod_lj_viscosity_PRA_1991py'
# rho and T taken from Ferrario et al. Phys Rev A, 44, 6936 (1991). Supposed to be LJ triple point
density = 0.8442
start_temperature = 3.0
target_temperature = 0.725
sc_output = 16
dt = 0.005
sr_list = [0.05, 0.10, 0.2, 0.316, 0.6, 0.8, 1.0, 1.41, 1.73, 2.0] # [0.02, 0.04, 0.8, 0.16]



direc = "../tests/reference_data"
conf_filename = direc + "/" + f'conf_LJ_N2048_rho{density:.4f}_{target_temperature:.4f}.h5'

gridsync = True

if run_NVT:
    # Setup configuration: FCC Lattice
    configuration = gp.Configuration(D=3, compute_flags={'stresses':True})
    configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=density)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=start_temperature)


    # Setup integrator to melt the crystal
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



configuration.simbox = gp.LeesEdwards(configuration.D, configuration.simbox.get_lengths())


# set the kinetic temperature to the exact value associated with the desired
# temperature since SLLOD uses an isokinetic thermostat
configuration.set_kinetic_temperature(target_temperature, ndofs=configuration.N*3-4) # remove one DOF due to constraint on total KE

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
totalStrain = 100.0
steps_per_block = 4096

sllod_results = []

for sr in sr_list:

    total_steps = int(totalStrain / (sr*dt)) + 1
    num_blocks = total_steps // steps_per_block + 1
    strain_transient = 1.0 # how much of the output to ignore (ie 1 corresponds to the first 100% of strain)
    num_steps_transient = int(strain_transient / (sr*dt) ) + 1

    integrator_SLLOD = gp.integrators.SLLOD(shear_rate=sr, dt=dt)
    # Setup runtime actions, i.e. actions performed during simulation of timeblocks
    runtime_actions = [gp.TrajectorySaver(include_simbox=True),
                       gp.MomentumReset(100),
                       gp.StressSaver(sc_output, compute_flags={'stresses':True}),
                       gp.ScalarSaver(sc_output)]


    #calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=1000)

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
        #calc_rdf.update()
        print(f'box-shift={box_shift:.4f}, strain = {box_shift/lengths[1]:.4f}')
    print(sim_SLLOD.summary())


    full_stress_tensor = gp.StressSaver.extract(sim_SLLOD.output)
    sxy = full_stress_tensor[:,0,1]

    times = np.arange(len(full_stress_tensor)) * sc_output *  dt
    strains = times * sr
    #stacked_output = np.column_stack((times, sxy))
    #np.savetxt('shear_run.txt', stacked_output, delimiter=' ', fmt='%f')


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

    sllod_results.append([sr, viscosity, error_on_visc])
#print("RESULTS")
#print(sllod_results)
sllod_results = np.array(sllod_results)

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
index_map={0:2, 1:3, 2:5, 3:7, 4:8, 5:9, 6:10, 7:11, 8:12, 9:13} # map the simulated items to the Ferrario list
chi_squared = 0.
num_items = len(sllod_results)
for idx in range(num_items):
    sr1 = sllod_results[idx, 0]
    v1 = sllod_results[idx, 1]
    dv1 = sllod_results[idx, 2]
    f_index = index_map[idx]
    sr2 = data_Ferrario_1991[f_index,0]
    v2 = data_Ferrario_1991[f_index,1]
    dv2 = data_Ferrario_1991[f_index,2]
    assert np.isclose(sr1, sr2)
    chi_squared += (v1-v2)**2 / (dv1**2+dv2**2)

critical_value_chi_sq = scipy.stats.chi2.ppf(0.95,df=num_items)
success = chi_squared < critical_value_chi_sq
print(f'Chi-squared = {chi_squared}; critical value={critical_value_chi_sq}')
if success:
    print('SUCCESS!')
else:
    print('FAIL')

title = f' Science test ({filename}.py): SUCCESS\n'
title += f'LJ fluid at triple point, N=2048, rho={density: .4f}, T={target_temperature:.4f} \n'
title += 'Reference: PRA v44 6936 (1991), doi.org/10.1103/PhysRevA.44.6936'
plt.title(title)

plt.errorbar(data_Ferrario_1991[:,0],data_Ferrario_1991[:,1],data_Ferrario_1991[:,2], label='Ferrario et al, Phys. Rev. A 1991')
plt.errorbar(sllod_results[:,0],sllod_results[:,1],sllod_results[:,2], label='Gamdpy')
plt.xlabel('Strain rate')
plt.ylabel('Viscosity')
plt.legend()
plt.show()


# Ferrario also used SLLOD to measure viscosity of the LJ fluid at different
# strain rates. Same density, temperature, N, time step. Cutoff slightly
# different: shifted potential with quadratic smoothing between 2.5 and 2.51.
# 10^7 time steps (my longest is SR 0.01 with 2 million). Also different
# thermostat for SLLOD
