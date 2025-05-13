

import numpy as np
import gamdpy as gp

# Sim. params 
rho, temperature = 1.0, 1.5
NVE = False  # If True -> k small
angle0, k = 2.0, 500.0
#rbcoef=[15.5000,  20.3050, -21.9170, -5.1150,  43.8340, -52.6070]
rbcoef=[.0, 50.0, .0, .0, .0, .0]

# Generate configuration with a FCC lattice
configuration = gp.Configuration(D=3, type_names=['CA', 'CT'])
configuration.make_lattice(gp.unit_cells.FCC, cells=(8, 8, 8), rho=rho)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=temperature)

# Setup topology  # could be read by something like configuration.topology.read('topol.top')
configuration.topology.add_molecule_name('Butane')
for n in range(0, configuration.N, 4):
    configuration.topology.bonds.append([n, n+1, 0])
    configuration.topology.bonds.append([n+1, n+2, 0])
    configuration.topology.bonds.append([n+2, n+3, 0])

    configuration.topology.angles.append([n, n+1, n+2, 0])
    configuration.topology.angles.append([n+1, n+2, n+3, 0])

    configuration.topology.dihedrals.append([n, n+1, n+2, n+3, 0])

    configuration.topology.molecules['Butane'].append([n, n+1, n+2, n+3])


# Make bond interactions
bond_potential = gp.harmonic_bond_function
bond_params = [[1.0, 1000.], ]
bonds = gp.Bonds(bond_potential, bond_params, configuration.topology.bonds)

# Make angle interactions
angle_potential = gp.cos_angle_function
angle_params = [[k, angle0],]
angles = gp.Angles(angle_potential, configuration.topology.angles, angle_params)

# Make dihedral interactions
dihedral_potential = gp.ryckbell_dihedral
dihedral_params = [rbcoef, ]
dihedrals = gp.Dihedrals(dihedral_potential, configuration.topology.dihedrals, dihedral_params)

# Exlusion list
exclusions = dihedrals.get_exclusions(configuration)


# Make pair potential
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Make integrator
if NVE:
    integrator = gp.integrators.NVE(dt=0.001)
else:
    integrator = gp.integrators.NVT(temperature=temperature, tau=0.1, dt=0.002)

# Compute plan
compute_plan = gp.get_default_compute_plan(configuration)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]

# Setup simulation
sim = gp.Simulation(configuration, [pair_pot, bonds, angles, dihedrals], integrator, runtime_actions,
                    num_timeblocks=10, steps_per_timeblock=256,
                    compute_plan=compute_plan, storage='memory')

angles_array, dihedrals_array = [], []
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))     
    angles_array.append( angles.get_angle(10, configuration) )
    dihedrals_array.append( dihedrals.get_dihedral(10, configuration) )

print(sim.summary()) 

columns = ['U', 'W', 'K',] 
data = np.array(gp.extract_scalars(sim.output, columns, first_block=1))
temp = 2.0/3.0*np.mean(data[2])/configuration.N
Etot = data[0] + data[2]
Etot_mean = np.mean(Etot)/configuration.N
Etot_std = np.std(Etot)/configuration.N

print("Temp:  %.2f  Etot: %.2e (%.2e)" % (temp,  Etot_mean, Etot_std))
print("Angle: %.2f (%.2f) " % (np.mean(angles_array), np.std(angles_array)))
print("Dihedral: %.2f (%.2f) " % (np.mean(dihedrals_array), np.std(dihedrals_array)))

print('\nFinal molecular potential energies: ')
molecular_energies = np.array( [ np.sum(configuration['U'][atoms])
                      for atoms in configuration.topology.molecules['Butane'] ])
print(molecular_energies[:15])

print(np.mean(molecular_energies), np.mean(configuration['U'])*4)
assert np.isclose(np.mean(molecular_energies), np.mean(configuration['U'])*4)
