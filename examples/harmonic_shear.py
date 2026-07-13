""" Example of a Simulation using gamdpy, using explicit blocks.

Simulation of a Lennard-Jones crystal in the NVT ensemble followed by shearing with SLLOD 
and Lees-Edwards boundary conditions using a harmonically varying shear rate

"""
import h5py
import numpy as np
import gamdpy as gp


temperature = 0.700
sc_output = 16
dt = 0.006
epsilon0 = 0.04
period = 200.
num_periods = 5.


with h5py.File('../tests/reference_data/conf_LJ_N2048_rho0.973_T0.700.h5', 'r') as fin:
    configuration = gp.Configuration.from_h5(fin, "configuration", compute_flags={'stresses':True})


configuration.simbox = gp.LeesEdwards(configuration.D, configuration.simbox.get_lengths())


# Setup pair potential: Single component 12-6 Lennard-Jones
pairfunc = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = gp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)


# The strain is a sine function with amplitude epsilon0
# Therefore the strain rate is a cosine with amplitude epsilon0*omega=epsilon0*2*pi/T

amplitude = epsilon0*2.*np.pi/period

strain_rate = gp.make_function_cos(period, amplitude, 0.)



integrator_SLLOD = gp.integrators.SLLOD(shear_rate=strain_rate, dt=dt)

# set the kinetic temperature to the exact value associated with the desired
# temperature since SLLOD uses an isokinetic thermostat
configuration.set_kinetic_temperature(temperature, ndofs=configuration.N*3-4) # remove one DOF due to constraint on total KE

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block

steps_per_block = 2048
total_steps = int(num_periods*period / dt)

num_blocks = total_steps // steps_per_block + 1
nominal_total_time = period*num_periods
actual_total_time = num_blocks * steps_per_block*dt
print(f'Nominal total time {nominal_total_time}')
print(f'Actual total time {actual_total_time}')


# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(include_simbox=True),
                   gp.MomentumReset(100),
                   gp.StressSaver(sc_output, compute_flags={'stresses':True}),
                   gp.ScalarSaver(sc_output)]


print(f'num_blocks={num_blocks}')
sim_SLLOD = gp.Simulation(configuration, pairpot, integrator_SLLOD, runtime_actions,
                          num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                          storage='memory')

# Run simulation one block at a time
for block in sim_SLLOD.run_timeblocks():
    print(sim_SLLOD.status(per_particle=True))
    configuration.simbox.copy_to_host()
    box_shift = configuration.simbox.box_shift
    lengths = configuration.simbox.get_lengths()
    print(f'box-shift={box_shift:.4f}, strain = {box_shift/lengths[1]:.4f}')
print(sim_SLLOD.summary())


U, K, W = gp.ScalarSaver.extract(sim_SLLOD.output, ['U', 'K', 'W'])
N = configuration.N
u, k = U/N,K/N

# alternative (newer way) to get the shear stress
full_stress_tensor = gp.StressSaver.extract(sim_SLLOD.output)
sxy = full_stress_tensor[:,0,1]

times = np.arange(len(u)) * sc_output *  dt
stacked_output = np.column_stack((times, u, k, sxy))
np.savetxt('shear_run.txt', stacked_output, delimiter=' ', fmt='%f')


