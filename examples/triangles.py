import numpy as np
import gamdpy as gp

gp.select_gpu()

# Simulation params 
rho, temperature = 0.3, .35 
filename = 'Data/triangles'
num_timeblocks = 128 * 1
steps_per_timeblock = 1024 * 2 

positions = [[0,   0,    ], 
             [1,   0,    ],  
             [0.5, 0.866 ], 
             [0.5, 0.300]]  # central atom,
particle_types = [1, 1, 2, 0]   
masses = [1, 1, 1, 1]           
                                                           
# Setup configuration: Single molecule first, then duplicate
top = gp.Topology(['Triangle', ])
top.bonds = [[0, 1, 0], [0, 2, 0], [1, 2, 0], 
             [3, 0, 1], [3, 1, 1], [3, 2, 1]]
top.molecules['Triangle'] = gp.molecules_from_bonds(top.bonds)

# This call creates the pdf "triangle.pdf" with a drawing of the molecule 
# Use block=True to visualize the molecule before running the simulation
gp.plot_molecule(top, positions, particle_types, filename="triangle.pdf", block=False)

configuration = gp.duplicate_molecule(top, positions, particle_types, masses, cells=(20, 20), safety_distance=4.0)
configuration.randomize_velocities(temperature=temperature)

#for i in range(0,configuration.N,4*30):
#    configuration.ptype[i+1] = 2 

print(f'Number of molecules: {len(configuration.topology.molecules["Triangle"])}')
print(f'Number of particles: {configuration.N}\n')

# Make bond interactions
bond_potential = gp.harmonic_bond_function
bond_params = [[1.0, 3000.], [0.566, 3000.]]
bonds = gp.Bonds(bond_potential, bond_params, configuration.topology.bonds)

# Exlusion list
#exclusions = dihedrals.get_exclusions(configuration)
exclusions = bonds.get_exclusions(configuration)

b = 0.7 # Gab between triangles 
c = b + 0.3 # Preferred distance between center particles
f = (1 + b**2)**.5
 
b /=  2**(1/6) 
c /=  2**(1/6) 
f /=  2**(1/6) 

d = 0.1 # relative strength of cross-binding (unlike particles)

# Make pair potential
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[c, c, c],
       [c, f, b],
       [c, b, f]]
eps = [[1, 1, 1],
       [1, 2, d],
       [1, d, 2],]
cut = [[2.5*c, 2.5*c, 2.5*c],
       [2.5*c, 1.0*f, 2.5*b],
       [2.5*c, 2.5*b, 1.0*f],]
#cut = 2.5*np.array(sig)
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Make integrator
integrator = gp.integrators.NVT_Langevin(temperature=temperature, alpha=0.4, dt=0.002, seed=23)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]

# Setup simulation
sim = gp.Simulation(configuration, [pair_pot, bonds,], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    storage='memory')

print('\nCompression and equilibration: ')
dump_filename = 'Data/dump_compress.lammps'
with open(dump_filename, 'w') as f:
    print(gp.configuration_to_lammps(sim.configuration, timestep=0), file=f)

initial_rho = configuration.N / configuration.get_volume()
for block in sim.run_timeblocks():
    volume = configuration.get_volume()
    N = configuration.N
    print(sim.status(per_particle=True), f'rho= {N/volume:.3}', end='\t')
    print(f'P= {(N*temperature + np.sum(configuration["W"]))/volume:.3}') # pV = NkT + W
    with open(dump_filename, 'a') as f:
        print(gp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

    # Scale configuration to get closer to final density, rho
    if block<sim.num_blocks/2:
        desired_rho = (block+1)/(sim.num_blocks/2)*(rho - initial_rho) + initial_rho
        configuration.atomic_scale(density=desired_rho)
        configuration.copy_to_device() # Since we altered configuration, we need to copy it back to device
print(sim.summary()) 
print(configuration)

sim = gp.Simulation(configuration, [pair_pot, bonds,], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    compute_plan=sim.compute_plan, storage=filename+'.h5')

print('\nProduction: ')
dump_filename = 'Data/dump.lammps'
with open(dump_filename, 'w') as f:
    print(gp.configuration_to_lammps(sim.configuration, timestep=0), file=f)

for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    with open(dump_filename, 'a') as f:
        print(gp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

print(sim.summary()) 
print(configuration)


print('\nAnalyse structure with:')
print('   python3 analyze_structure.py Data/molecules')

print('\nAnalyze dynamics with:')
print('   python3 analyze_dynamics.py Data/molecules')

print('\nVisualize simulation in ovito with:')
print('   ovito Data/dump.lammps')

#print('\nVisualize simulation in VMD with:')
#print('   vmd -lammpstrj Data/dump.lammps')
