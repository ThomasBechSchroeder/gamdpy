""" Minimal example of a constant NpT simulation

Simulation of a Lennard-Jones liquid in the NPT ensemble.
After equilibration, the simulation runs and calculates the mean potential energy, 
pressure, density and isothermal compressibility.

"""

import gamdpy as gp

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3, compute_flags={'Vol':True})
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.7543)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=2.0)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup NPT integrator
target_temperature = 2.0  # target temperature for barostat
target_pressure = 4.70  # target pressure for barostat
integrator = gp.integrators.NPT_Atomic(temperature=target_temperature, 
                                       tau=0.4,
                                       pressure=target_pressure, 
                                       tau_p=5,
                                       dt=0.001)

# Set up runtime actions
runtime_actions = [gp.RestartSaver(),
                   gp.TrajectorySaver(), 
                   gp.ScalarSaver(64),
                   gp.MomentumReset(100)]


# NPT Simulation (Equilibration)
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=16, steps_per_timeblock=1 * 1024,
                    storage='memory')
print("Equilibration")
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True), f"rho= {configuration.N/configuration.get_volume():6.3f}")
print(sim.summary())


# NPT Simulation (Production)
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=16, steps_per_timeblock=1 * 1024,
                    storage='LJ_p4.70_T2.0.h5')  # LJ_p4.70_T2.0.h5 Data/LJ_p4.70_T2.0_toread.h5
print("Production")
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True), f"rho= {configuration.N/configuration.get_volume():6.3f}")
print(sim.summary())

# Thermodynamic properties
N, D = configuration.N, configuration.D
fluctuations = gp.ScalarSaver.extract_as_dict(sim.output, first_block=0)
dof = D * N - D
data = gp.tools.thermodynamics_NpT(N, dof, **fluctuations, T_ext=target_temperature, p_ext=target_pressure)
for key in data:
    print(f'{key:>32} = {data[key]:10.5f}')
