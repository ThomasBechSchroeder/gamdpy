""" Example of performing several simulation in one go using gamdpy.

An isomorph is traced out using the gamma method. The script demomstrates
the possibility of keeping the output of the simulation in memory (storage='memory').
This is usefull when a lot of short simulations are performed.

To plot the results do: 
python plot_isomorph_dynamics.pdf
python plot_isomorph_rdf.pdf

For a simpler script performing multiple simulations, see isochore.py

"""

import matplotlib.pyplot as plt

import gamdpy as gp

# Setup pair potential.
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

T = 0.8

# Setup fcc configuration
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.84)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)
  
# Setup integrators
integrator1 = gp.integrators.NVT(temperature=T, tau=0.2, dt=0.0025)
integrator2 = gp.integrators.NVT(temperature=T, tau=0.2, dt=0.0025)

runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(1),
                   gp.MomentumReset(100)]

# Setup Simulations
sim1 = gp.Simulation(configuration, pair_pot, integrator1, runtime_actions,
                     num_timeblocks=4, steps_per_timeblock=512,
                     storage='memory')

print(configuration['r'][1])
print('Integrator1, Equilibration:', end='\t')
for block in sim1.run_timeblocks():
    pass
print(sim1.status(per_particle=True))
U1, K1 = gp.extract_scalars(sim1.output, ['U', 'K'], first_block=0)
E1 = U1 + K1

print('Integrator1, Production:', end='\t')
for block in sim1.run_timeblocks():
    pass
print(sim1.status(per_particle=True))
U2, K2 = gp.extract_scalars(sim1.output, ['U', 'K'], first_block=0)
E2 = U2 + K2 

sim2 = gp.Simulation(configuration, pair_pot, integrator2, runtime_actions,
                     num_timeblocks=4, steps_per_timeblock=512,
                     storage='memory')

print(configuration['r'][1])
                     
print('Integrator2, Production:', end='\t')
for block in sim2.run_timeblocks():
    pass
print(sim2.status(per_particle=True))

U3, K3 = gp.extract_scalars(sim2.output, ['U', 'K'], first_block=0)
E3 = U3 + K3 


plt.plot(U1, '.-', label='Integrator1, Equilibration')
plt.plot(U2, '.-', label='Integrator1, Production')
plt.plot(U3, '.-', label='Integrator2, Production')
plt.legend()
if __name__ == "__main__":
    plt.show()
