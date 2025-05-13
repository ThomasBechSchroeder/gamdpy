""" The classical XY model of rotators on a 2D square lattice

The XY model describes a system of planar rotators
that can take any angle θ in the range [-π, π].
The energy of the system arises from interactions
between neighboring spins, which favor alignment.

- A square lattice is defined with dimensions L×L.
- Each rotator is assigned an initial angle given as a position, using
  an orthorhombic cell with periodic boundaries over the interval [-π, π].
- "Bonds" are generated between a given rotator and its nearest neighbors
  to the right and below on the lattice.
- The function `neighbour_potential` defines the interactions of bonds
"""
import numpy as np
import matplotlib.pyplot as plt
import gamdpy as gp
from math import sin, cos, pi

# Setup configuration
L = 128
N = L*L
configuration = gp.Configuration(D=1, N=N)
configuration['r'] = 2*pi*np.random.random(N).reshape(N,1)  # "positions" are theta angles of the rotators
configuration['m'] = np.ones(N)  # "masses" are moment of inertia
configuration.simbox = gp.Orthorhombic(1, np.array([2*pi]))
configuration.randomize_velocities(temperature=100)

# Generate bonds to neighbours in square lattice with periodic boundaries
neighbour_bonds = []
for this in range(N):
    row= this // L
    col = this % L
    neighbour_bonds.append([this, ((row + 1) % L) * L + col, 0])
    neighbour_bonds.append([this, row * L + ((col + 1) % L), 0])

# Define "bond" interactions with neighbours in lattice
def neighbour_potential(delta_theta: float, params: np.ndarray) -> tuple:
    """ Neighbour interactions """
    eps = params[0]
    x = delta_theta
    u = -cos(x)
    s = -1.0 if x == 0 else -sin(x)/x  # Force multiplier, -u'(r)/r
    curvature = cos(x)
    return eps*u, eps*s, eps*curvature
neighbour_bond_potential = neighbour_potential
neighbour_bond_params = [[1.0], ]
neighbour_bond_interaction = gp.Bonds(neighbour_bond_potential, neighbour_bond_params, neighbour_bonds)

# Setup integrator
integrator = gp.integrators.NVT_Langevin(temperature=0.4, alpha=0.4, dt=0.01, seed=2025)

# Setup runtime actions
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]

# Compute plan
compute_plan = gp.get_default_compute_plan(configuration)

# Setup simulation
interactions = [neighbour_bond_interaction, ]
sim = gp.Simulation(configuration, interactions, integrator, runtime_actions,
                    num_timeblocks=16, steps_per_timeblock=2048,
                    compute_plan=compute_plan, storage='memory')

# Run simulation
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
print(sim.summary())

# Make figure with theta angles as colors
plt.figure()
grid = np.array(configuration['r']).reshape(L, L)
plt.imshow(grid, cmap='hsv', vmin=-pi, vmax=pi, origin='lower')
plt.colorbar()
plt.show()
