""" A rubber cube modeled as particles connected by springs """

from itertools import product
from math import pi, sin, cos

import numpy as np
import matplotlib.pyplot as plt

import gamdpy as gp

# A rotated cube of particles
lattice_length = 8 # Number of particles in each direction
cm = np.array([0.0, 0.0, 0.0])  # Initial position of center of mass
theta_z = pi/12  # 1st rotation around z-axis
theta_x = pi/24  # 2nd rotation around x-axis
cube = np.array([(x, y, z) for x, y, z in product(range(lattice_length), repeat=3)])
center = cube.mean(axis=0)
rot_matrix_z = np.array([
    [cos(theta_z),  sin(theta_z), 0],
    [-sin(theta_z), cos(theta_z), 0],
    [0,             0,            1]
])
rot_matrix_x = np.array([
    [1,             0,            0],
    [0, cos(theta_z),  sin(theta_z)],
    [0, -sin(theta_z), cos(theta_z)],
])
N = lattice_length**3
cube = (cube - center) @ rot_matrix_z @ rot_matrix_x + cm
configuration = gp.Configuration(D=3, N=N)
configuration['r'] = cube
configuration['m'] = 1.0
box_length = 64
configuration.simbox = gp.Orthorhombic(3, [box_length, box_length, box_length])
configuration.randomize_velocities(temperature=0.001)
# Compute plan
compute_plan = {'pb': 16, 'tp': 4, 'skin': 2.0, 'UtilizeNIII': False, 'gridsync': False, 'nblist': 'N squared'}

# Create bonds
bond_potential = gp.harmonic_bond_function
bond_params = [
    [1.0, 100.0],
    [2**0.5, 10.0]
]
bond_directions =[
    # (dx, dy, dz), bond_type
    ((1, 0, 0), 0),
    ((0, 1, 0), 0),
    ((0, 0, 1), 0),
    ((0, 1, 1), 1),
    ((1, 0, 1), 1),
    ((1, 1, 0), 1)
]
L = lattice_length
def xyz2idx(x, y, z):
    return x + y*L + z*L**2
neighbour_bonds = []
for x, y, z in product(range(L), repeat=3):
    this = xyz2idx(x, y, z)
    for (dx, dy, dz), bond_type in bond_directions:
        if x + dx < L and y + dy < L and z + dz < L:
            that = xyz2idx(x+dx, y+dy, z+dz)
            neighbour_bonds.append([this, that, bond_type])
bonds = gp.Bonds(bond_potential, bond_params, neighbour_bonds)

# Create a 3D figure showing bonds
plot = False
if plot:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pos = configuration['r']
    for bond in neighbour_bonds:
        n, m, _ = bond
        xs = [pos[n, 0], pos[m, 0]]
        ys = [pos[n, 1], pos[m, 1]]
        zs = [pos[n, 2], pos[m, 2]]
        ax.plot(xs, ys, zs, 'k-', linewidth=1)
    ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], c=cube[:, 2])
    plt.show()

# Add two smooth walls
wall_distance = box_length/2
walls = gp.interactions.Planar(
    potential=gp.harmonic_repulsion,
    params=[[100.0, 1.0], [100.0, 1.0]],
    indices=[[n, 0] for n in range(N)] + [[n, 1] for n in range(N)],  # All particles feel both walls
    normal_vectors=[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
    points=[[0.0, -wall_distance/2, 0], [0.0, wall_distance/2, 0]]
)

# Add gravity
mg = 0.0005
potential_gravity = gp.make_IPL_n(-1)
gravity = gp.interactions.Planar(
    potential=potential_gravity,
    params= [[mg, 10*wall_distance]],
    indices= [[n, 0] for n in range(N)],   # All particles feel the gravity
    normal_vectors= [[0,1,0], ],
    points= [[0, -wall_distance/2.0, 0] ]  # Defining 0 for potential energy on lower wall
)

# Setup simulation
integrator = gp.integrators.NVE(dt=0.01)
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(steps_between_output=1)]
interactions = [bonds, walls, gravity]
sim = gp.Simulation(configuration, interactions, integrator, runtime_actions,
                    num_timeblocks=64, steps_per_timeblock=1024,
                    storage='memory')

# Run simulation and save as lammps trajectory
dump_filename = 'Data/rubber_cube.lammps'
with open(dump_filename, 'w') as f:
    print(gp.configuration_to_lammps(sim.configuration, timestep=0), file=f)

for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    with open(dump_filename, 'a') as f:
        print(gp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

print(sim.summary())
