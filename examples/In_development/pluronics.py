import numpy as np
import gamdpy as gp

gp.select_gpu()

# Simulation params 
rho, temperature  = 0.38, 1.2
N_A, N_B = 12, 8  # A - ... - A - B - ... - B - A - ... - A 

particles_per_molecule = 2*N_A + N_B
filename = 'Data/pluronics'
num_timeblocks = 64
steps_per_timeblock = 1 * 1024 # 

positions = []
particle_types = []
masses = []

bond_length = 0.8

# A particles, 1. arm
for i in range(N_A):
    positions.append( [ i*bond_length, (i%2)*.1, 0. ] ) # x, y, z for this particle
    particle_types.append( 0 )
    masses.append( 1.0 )  

# B particles
for i in range(N_B):
    positions.append( [ (i+N_A)*bond_length, 0, ((i+1)%2)*.1 ] ) # x, y, z for this particle
    particle_types.append( 1 )
    masses.append( 1.0 )  

# A particles, 2. arm
for i in range(N_A):
    positions.append( [ (i+N_A+N_B)*bond_length, 0, ((i+1)%2)*.1, ] ) # x, y, z for this particle
    particle_types.append( 0 )
    masses.append( 1.0 )  

# Setup configuration: Single molecule first, then duplicate
top = gp.Topology(['MyMolecule', ])
top.bonds = gp.bonds_from_positions(positions, cut_off=bond_length+.1, bond_type=0)
top.molecules['MyMolecule'] = gp.molecules_from_bonds(top.bonds)

print('Initial Positions:')
for position in positions:
    print('\t\t', position)
print('Particle types:\t', particle_types)
print('Bonds:         \t', top.bonds)
print()

# This call creates the pdf "molecule.pdf" with a drawing of the molecule 
# Use block=True to visualize the molecule before running the simulation
gp.plot_molecule(top, positions, particle_types, filename="molecule.pdf", block=False)

configuration = gp.duplicate_molecule(top, positions, particle_types, masses, cells=(4, 4, 4), safety_distance=2.0)
configuration.randomize_velocities(temperature=temperature)

print(f'Number of molecules: {len(configuration.topology.molecules["MyMolecule"])}')
print(f'Number of particles: {configuration.N}\n')

# Make bond interactions
bond_potential = gp.harmonic_bond_function
bond_params = [[0.8, 1000.], ]
bonds = gp.Bonds(bond_potential, bond_params, configuration.topology.bonds)

# Exlusion list
exclusions = bonds.get_exclusions(configuration)

# Make pair potential
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[1.00, 1.00,],
       [1.00, 1.00,],]
eps = [[1.00, 1.00],
       [1.00, 1.00],]
cut = [[1.12, 1.12,],
       [1.12, 2.50,],]

pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Make integrator
dt = 0.005
integrator = gp.integrators.NVT_Langevin(temperature=temperature, alpha=0.1, dt=0.004, seed=234)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]

# Setup simulation
sim = gp.Simulation(configuration, [pair_pot, bonds,], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    storage='memory')

print('\nCompression and equilibration: ')
dump_filename = 'Data/pluronics_compress.lammps'
with open(dump_filename, 'w') as f:
    print(gp.configuration_to_lammps(sim.configuration, timestep=0), file=f)

initial_rho = configuration.N / configuration.get_volume()
for block in sim.run_timeblocks():
    volume = configuration.get_volume()
    N = configuration.N
    current_rho = N/volume
    print(sim.status(per_particle=True), f'rho= {current_rho:.3}', end='\t')
    print(f'P= {(N*temperature + np.sum(configuration["W"]))/volume:.3}') # pV = NkT + W
    with open(dump_filename, 'a') as f:
        print(gp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

    # Scale configuration to get closer to final density, rho
    if block<sim.num_blocks / 2:
        desired_rho = (block+1 )/(sim.num_blocks/2)*(rho - initial_rho) + initial_rho
        if desired_rho > 1.2*current_rho:
            desired_rho = 1.2*current_rho
        configuration.atomic_scale(density=desired_rho)
        configuration.copy_to_device() # Since we altered configuration, we need to copy it back to device
print(sim.summary()) 
print(configuration)

sim = gp.Simulation(configuration, [pair_pot, bonds, ], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    compute_plan=sim.compute_plan, storage=filename+'.h5')

print('\nProduction: ')
dump_filename = 'Data/pluronics_dump.lammps'
with open(dump_filename, 'w') as f:
    print(gp.configuration_to_lammps(sim.configuration, timestep=0), file=f)

for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    with open(dump_filename, 'a') as f:
        print(gp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

print(sim.summary()) 
print(configuration)

print('\nAnalyse structure with:')
print('   python3 analyze_structure.py Data/pluronics')

print('\nAnalyze dynamics with:')
print('   python3 analyze_dynamics.py Data/pluronics')

print('\nVisualize simulation in ovito with:')
print('   ovito Data/pluronics_dump.lammps')

#print('\nVisualize simulation in VMD with:')
#print('   vmd -lammpstrj Data/dump.lammps')
