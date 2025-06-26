'''
    Example of water simulation: Starts from dilute setup which is compressed. The production run.
    Model: SPC/Fw (Wu et al. J. Chem. Phys. 124:024505 (2006) 
'''

import numpy as np
import gamdpy as gp

# Sim. params 
nmols = 1000
rho0= 0.1
rho_desired, temperature = 3.15, 3.9

qH,qO = 10.783, -21.566
mH,mO = 1.0/16.0, 1.00
bond0, kspring=0.316, 68000
angle0, kangle=1.97,490

dt = 0.0005
alpha = 10.0

# Atom positions; H-O-H
r0=[[0.00, 0.184, 0.0],
    [0.26, 0.0, 0.0],
    [0.53, 0.184, 0.0]]
mass=[mH, mO, mH]
types=[0, 1, 0]

top = gp.Topology(['water', ])
top.bonds = gp.bonds_from_positions(r0, cut_off=0.5, bond_type=0)
top.angles = gp.angles_from_bonds(top.bonds, angle_type=0)
top.molecules['water'] = gp.molecules_from_bonds(top.bonds)

dict_this_mol = {"positions" : r0,
                 "particle_types" : types,
                 "masses" : mass,
                 "topology" : top}

configuration = gp.replicate_molecules([dict_this_mol], [nmols], safety_distance=2.0)
configuration.randomize_velocities(temperature=temperature)

# Make bonds
bond_potential = gp.harmonic_bond_function
bond_params = [[bond0, kspring], ]
bonds = gp.Bonds(bond_potential, bond_params, configuration.topology.bonds)

# Angles
angle_potential = gp.cos_angle_function
angle_params = [[kangle, angle0],]
angles = gp.Angles(angle_potential, configuration.topology.angles, angle_params)

# Angle exclusions
exclusion = np.zeros( (configuration.N, 20) , dtype=np.int32)
for n in range(0, configuration.N, 3):
    exclusion[n, 0], exclusion[n, 1], exclusion[n, -1] = n+1, n+2, 2
    exclusion[n+1, 0], exclusion[n+1, 1], exclusion[n+1, -1] = n, n+2, 2
    exclusion[n+2, 0], exclusion[n+2, 1], exclusion[n+2, -1] = n, n+1, 2
    

# Make pair potential
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_SF)
sig = [[0.0, 0.0], 
       [0.0, 1.0]]
eps = [[0.0, 0.0], 
       [0.0, 1.0]]
charge = [[qH*qH, qH*qO], 
          [qO*qH, qO*qO]]
cut = np.ones( (2,2) )*2.9

pair_pot = gp.PairPotential(pair_func, params=[sig, eps, charge, cut], exclusions=exclusion, max_num_nbs=1000)

# Make integrator
integrator = gp.integrators.NVT(temperature=temperature, tau=0.1, dt=dt)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.MomentumReset(100), ]


# Eq. setup simulation
sim = gp.Simulation(configuration, [pair_pot, bonds, angles], integrator, runtime_actions,
                    num_timeblocks=4000, steps_per_timeblock=64, storage='memory')

npart = configuration.N
for block in sim.run_timeblocks():
    rho = npart/configuration.simbox.get_volume()
    prefac = np.exp(-alpha*rho/rho_desired) + 1.0  
    if prefac > 1.01:
        prefac = 1.01

    if rho < rho_desired:
        rho = prefac*rho
        configuration.atomic_scale(density=rho)
        configuration.copy_to_device()
    
print("Eq. done", rho)
print(sim.status(per_particle=True))

runtime_actions = [gp.MomentumReset(100), gp.TrajectorySaver()]

sim = gp.Simulation(configuration, [pair_pot, bonds, angles], integrator, runtime_actions,
                    num_timeblocks=100, steps_per_timeblock=512, storage='water.h5')

for block in sim.run_timeblocks():
    print("\r", block, "out of ", sim.num_blocks, end=" ")  


