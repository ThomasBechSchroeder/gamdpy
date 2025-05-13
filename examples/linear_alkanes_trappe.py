import numpy as np
import gamdpy as gp

def trappe_ua_molecule(molecule_name, positions, particle_types, type_names):
    """
    Generate a molecule and associated interactions using the TraPPe united atom forcefield
    (Not correct potentials or parameters yet!) 
    """

    trappe_known_types = ['CH2', 'CH3']
    trappe_masses = {'CH2':1.0, 'CH3':1.1}  # Should be read from a configuration-file

    for i in range(len(particle_types)):
        assert type_names[particle_types[i]] in trappe_known_types

    top = gp.Topology(molecule_names=[molecule_name, ])
    top.bonds = gp.bonds_from_positions(positions, cut_off=1.1, bond_type=0)
    top.angles = gp.angles_from_bonds(top.bonds, angle_type=0)
    top.dihedrals = gp.dihedrals_from_angles(top.angles, dihedral_type=0)
    top.molecules[molecule_name] = gp.molecules_from_bonds(top.bonds)

    molecule = {}
    molecule['top'] = top
    molecule['positions'] = positions # Need (deep-) copy?
    molecule['particle_types'] = particle_types
    molecule['type_names'] =  type_names
    molecule['masses'] = [ trappe_masses[type_names[particle_type]] for particle_type in particle_types]

    return molecule
    
def trappe_ua_interactions(configuration, type_names):
    interactions = {} 

    # Make bond interactions # Should rund through list at update bond-type accordording to atom-types
    bond_potential = gp.harmonic_bond_function
    bond_params = [[0.8, 1000.], ]
    interactions['bonds'] = gp.Bonds(bond_potential, bond_params, configuration.topology.bonds)

    # Make angle interactions
    angle_potential = gp.cos_angle_function
    angle0, k = 2.0, 500.0
    interactions['angles'] = gp.Angles(angle_potential, configuration.topology.angles, parameters=[[k, angle0],])

    # Make dihedral interactions
    rbcoef=[.0, 5.0, .0, .0, .0, .0]
    dihedral_potential = gp.ryckbell_dihedral
    interactions['dihedrals'] = gp.Dihedrals(dihedral_potential, configuration.topology.dihedrals, parameters=[rbcoef, ])

    # Exlusion list
    exclusions = interactions['dihedrals'].get_exclusions(configuration)

    # Make pair potential
    pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
    sig = [[1.00, 0.80],
           [0.80, 0.88]]
    eps = [[1.00, 1.50],
           [1.50, 0.50]]
    cut = np.array(sig)*2.5
    interactions['pair_pot'] = gp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

    return interactions
    
# Simulation parameters
rho, temperature = 0.9, 1.5
alkane_length = 8 
filename = 'Data/linear_alkanes'
num_timeblocks = 64 
steps_per_timeblock = 1 * 1024

# Generate one molecule. Could be read from file
positions = []
particle_types = []

for i in range(alkane_length):
    positions.append( [ i*1.0, (i%2)*.1, 0. ] ) # x, y, z for this particle
    particle_types.append( 0 )
# End atoms:
particle_types[0] = 1
particle_types[1] = 1

molecule = trappe_ua_molecule('alkane', positions, particle_types, type_names=['CH2', 'CH3'])
print(molecule)

configuration = gp.duplicate_molecule(molecule['top'], molecule['positions'], molecule['particle_types'],
                                      molecule['masses'], cells=(6, 6, 6), safety_distance=2.0)
configuration.randomize_velocities(temperature=temperature)

print(f'Number of molecules: {len(configuration.topology.molecules["alkane"])}')
print(f'Number of particles: {configuration.N}\n')

interactions = trappe_ua_interactions(configuration, type_names=['CH2', 'CH3'])

# Make integrator
integrator = gp.integrators.NVT(temperature=temperature, tau=0.1, dt=0.002)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]

# Setup simulation
sim = gp.Simulation(configuration, list(interactions.values()), integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    storage='memory')

print('\nCompression and equilibration: ')
dump_filename = 'Data/alkanes_compress.lammps'
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

sim = gp.Simulation(configuration, list(interactions.values()), integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    compute_plan=sim.compute_plan, storage=filename+'.h5')

print('\nProduction: ')
dump_filename = 'Data/alkanes_dump.lammps'
with open(dump_filename, 'w') as f:
    print(gp.configuration_to_lammps(sim.configuration, timestep=0), file=f)

for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    with open(dump_filename, 'a') as f:
        print(gp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

print(sim.summary()) 
print(configuration)

print('\nAnalyse structure with:')
print('   python3 analyze_structure.py Data/alkanes')

print('\nAnalyze dynamics with:')
print('   python3 analyze_dynamics.py Data/alkanes')

print('\nVisualize simulation in ovito with:')
print('   ovito Data/alkanes_dump.lammps')

#print('\nVisualize simulation in VMD with:')
#print('   vmd -lammpstrj Data/dump.lammps')
