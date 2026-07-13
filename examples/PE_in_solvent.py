
import math
import numpy as np
import gamdpy as gp
import numba

gp.select_gpu()

# This script matches the polyethylene model simulated in the LAMMSP tutorial that can be found here:
# https://download.lammps.org/tutorial/hands-on-part1/LAMMPS_Hands-on.pdf
# We create our system from scratch and equilibrate, rather than reading the same configuration file, so only statistical comparison is possible.

# In the tutorial they use a softer repulsion than the usual 12-6 Lennard-Jones, namely LJ 9-6, so we need to define that here.

def LJ_9_6(dist, params):
    """ A softer 9-6 Lennard-Jone potential, as used in LAMMPS, see
https://www.afs.enea.it/software/lammps/doc19/html/pair_sdk.html
(note on https://docs.lammps.org/pair_lj96.html the coefficient seems to be wrong, namely 4 as in LJ12-6 where it is 27/4 on the other site)
    Parameters the same as ordinary LJ. Has been tested by checking energy conservation in an NVE run, but the second derivative has not been
    tested.
    """
    sigma = params[0]
    epsilon = params[1]
    OneOdist = numba.float32(1.0) / dist
    sigmaOdist = sigma * OneOdist

    u = numba.float32(27.0/4.0) * epsilon * (sigmaOdist ** 9 - sigmaOdist ** 6)
    s = numba.float32(81.0/2.0) * epsilon * (numba.float32(1.5) * sigmaOdist ** 9 - sigmaOdist ** 6) * OneOdist ** 2
    umm = numba.float32(81.0/2.0) * epsilon * (
                numba.float32(15.0) * sigmaOdist ** 9 - numba.float32(7.0) * sigmaOdist ** 6) * OneOdist ** 2
    return u, s, umm  # U(r), s == -U'(r)/r, U''(r)



# Regarding units: the internal units are kcal/mol, Angstrom, and amu (Dalton)
# But we want to specify the temperature in K and the timestep in fs. To do this we unit
# gamdpy's conversion_factors function which returns a dictionary allowing unit conversions.

cf = gp.conversion_factors(unit_length_in_Angstrom=1.0, unit_energy_in_kcal_per_mol=1.0, unit_mass_in_u=1.0)




# Simulation params 
temperature_in_K = 300.
temperature = temperature_in_K / cf['K']
rho = 0.0066942767
timestep_in_fs = 5.0
timestep = timestep_in_fs / cf['fs']
filename = 'Data/PE'
num_timeblocks_equilibration = 64
num_timeblocks_production = 64
steps_per_timeblock = 1 * 1024

# first construct a chain of length 20, with end-beads having a different type to the rest
chain_length = 20
bond_lengths = [3.65, 3.64]
bond_strengths = [6.160*2., 6.160*2.]
pos_chain = []
types_chain = []
masses_chain = []
for i in range(chain_length):
    pos_chain.append( [i*bond_lengths[1], (i%2)*0.1*bond_lengths[1], 0.] ) # ignoring the slight difference between the end-center and center-center bond lengths
    types_chain.append(0 if i in [0, chain_length-1] else 1)
    masses_chain.append(43.089 if i in [0, chain_length-1] else 42.081)
    

top_chain = gp.Topology(['PE', ])
top_chain.bonds = gp.bonds_from_positions(pos_chain, cut_off=1.01*max(bond_lengths), bond_type=1)
# set the bond type at the ends to be zero
top_chain.bonds[0][2] = 0
top_chain.bonds[-1][2] = 0

top_chain.angles = gp.angles_from_bonds(top_chain.bonds, angle_type=1)
# set the angle types at the ends to be zero
top_chain.angles[0][2] = 0
top_chain.angles[-1][2] = 0


top_chain.molecules['PE'] = gp.molecules_from_bonds(top_chain.bonds)

mol_dict_PE = {"positions": pos_chain,
               "particle_types" : types_chain,
               "masses": masses_chain,
               "topology" : top_chain
               }

top_solvent = gp.Topology(['solvent'],)
top_solvent.bonds = []
top_solvent.angles = []
top_solvent.dihedrals = []
top_solvent.molecules['solvent'] = [[0]]


mol_dict_solvent = {"positions": [[0., 0., 0.]],
                    "particle_types": [2],
                    "masses": [43.089],
                    "topology": top_solvent}



configuration = gp.replicate_molecules([mol_dict_PE, mol_dict_solvent], [1, 1980], safety_distance=5.0)
configuration.randomize_velocities(temperature=temperature)


print(f'Number of molecules: {len(configuration.topology.molecules[f"PE"])}, {len(configuration.topology.molecules[f"solvent"])}')
print(f'Number of particles: {configuration.N}\n')

# Make bond interactions
bond_potential = gp.harmonic_bond_function
bond_params = [[bond_lengths[0], bond_strengths[0]], [bond_lengths[1], bond_strengths[1]]]
bonds = gp.Bonds(bond_potential, configuration.topology.bonds, bond_params)

# Make angle interactions
angle0, k0 = 175.0*math.pi/180., 1.190 * 2. # the 2 is to match LAMMPS which doesn't have an explicit factor 1/2
angle1, k1 = 173.0*math.pi/180., 1.190 * 2.

#angle_potential = gp.harmonic_angle_function
angle_potential = gp.make_harmonic_angle_function() # uses default SMALL=1.e-6 to deal with dividing by small values of sin(theta), other values can be passed here.
angles = gp.Angles(angle_potential, configuration.topology.angles, angle_parameters=[[angle0, k0], [angle1, k1]])


# Exlusion list
exclusions = angles.get_exclusions(configuration)

# Make pair potential
pair_func = gp.apply_shifted_potential_cutoff(LJ_9_6)
#pair_func = gp.apply_shifted_force_cutoff(LJ_9_6)
#pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[4.5850, 4.5455, 4.5850],
       [4.5455, 4.5060, 4.5455],
       [4.5850, 4.5455, 4.5850]]

eps = [[0.4690, 0.4440, 0.4690 ],
       [0.4440, 0.4200, 0.4440],
       [0.4690, 0.4440, 0.4690]]

Rc = 15.0
cut = [[Rc, Rc, Rc], [Rc, Rc, Rc], [Rc, Rc, Rc]]

pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Make integrator
integrator = gp.integrators.NVT(temperature=temperature, tau=200.*timestep, dt=timestep)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.RestartSaver(),
                   gp.TrajectorySaver(),
                   gp.ScalarSaver(1),
                   gp.MomentumReset(100)]

# Setup simulation
sim = gp.Simulation(configuration, [pair_pot, bonds, angles], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks_equilibration, steps_per_timeblock=steps_per_timeblock,
                    storage=filename+'_compress.h5')


print('\nCompression and equilibration: ')
initial_rho = configuration.N / configuration.get_volume()
for block in sim.run_timeblocks():
    volume = configuration.get_volume()
    N = configuration.N
    current_rho = N/volume
    print(sim.status(per_particle=True), f'rho= {current_rho:.3}', end='\t')
    print(f'P= {(N*temperature + np.sum(configuration["W"]))/volume:.3}') # pV = NkT + W
    
    # Scale configuration to get closer to final density, rho
    if block<sim.num_blocks/2:
        desired_rho = (block+1)/(sim.num_blocks/2)*(rho - initial_rho) + initial_rho
        if desired_rho > 1.5*current_rho:
            desired_rho = 1.5*current_rho 
        configuration.atomic_scale(density=desired_rho)
        configuration.copy_to_device() # Since we altered configuration, we need to copy it back to device
print(sim.summary()) 
print(configuration)



sim = gp.Simulation(configuration, [pair_pot, bonds, angles], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks_production, steps_per_timeblock=steps_per_timeblock,
                    compute_plan=sim.compute_plan, storage=filename+'.h5')

print('\nProduction: ')
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))

print(sim.summary()) 
print(configuration)

print('\nTo visualize in ovito (if installed):')
print(f'python3 visualize.py {filename}.h5')
