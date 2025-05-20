import sys
import numpy as np
import gamdpy as gp
from gamdpy.integrators import nvt_nh
# from gamdpy.interactions import pair, bond, planar
import numba
from numba import cuda
import pandas as pd
import matplotlib.pyplot as plt
import math

include_springs = 'springs' in sys.argv
include_walls = 'walls' in sys.argv
include_gravity = 'gravity' in sys.argv
include_KABLJ = 'KABLJ' in sys.argv

if include_gravity:
    assert include_walls, 'Do not do gravity without walls!!!'

rho = 0.85
wall_dist = 6.31 # Ingebrigtsen & Dyre (2014)
wall_dimension = 2
nxy, nz = 8, 4

# Generate configuration with a FCC lattice (higher rho, to make room for walls)
c1 = gp.Configuration(D=3)
c1.make_lattice(gp.unit_cells.FCC, cells=[nxy, nxy, nz], rho=1.5)
c1['m'] = 1.0
c1.randomize_velocities(temperature=1.44)

# Adjust the simbox according to the desired setup
current_lengths = c1.simbox.get_lengths()
Lxy = (c1.N/wall_dist/rho)**0.5
new_lengths = np.ones(3, dtype=np.float32) * Lxy
#c1.simbox.lengths[:] = Lxy
if include_gravity:
    wall_dist *= 2  # Double wall distance, so effect of gravity can be seen 
if include_walls:
    #c1.simbox.lengths[wall_dimension] = wall_dist + 4 # Make box bigger to avoid walls working through PBC
    new_lengths[wall_dimension] = wall_dist + 4 # Make box bigger to avoid walls working through PBC
else:
    #c1.simbox.lengths[wall_dimension] = wall_dist #
    new_lengths[wall_dimension] = wall_dist
if include_gravity:
    #c1.simbox.lengths[wall_dimension] *= 10.  # Make box even bigger to avoid weird PBC effects
    new_lengths[wall_dimension] *= 10.  # Make box even bigger to avoid weird PBC effects
if include_KABLJ:
    c1.ptype[np.arange(0,c1.N,4)] = 1    # 3:1 mixture

c1.simbox.scale(new_lengths/current_lengths)

c1.copy_to_device()

print('simbox: ', c1.simbox.get_lengths())
if include_walls:
    print('wall_distance: ', wall_dist)

compute_plan = gp.get_default_compute_plan(c1)
 
# Setup bond interactions (This is the bare-bones way - It should be possible to setup and replicate molecules)
if include_springs:
    bond_potential = gp.harmonic_bond_function
    potential_params_list = [[1.12, 1000.], [1.0, 1000.], [1.12, 1000.]]
    fourth = np.arange(0,c1.N,4)
    bond_particles_list = [np.array((fourth, fourth+1)).T, np.array((fourth+1, fourth+2)).T, np.array((fourth+2, fourth+3)).T] 
    bonds = gp.setup_bonds(c1, bond_potential, potential_params_list, bond_particles_list, compute_plan, verbose=True)
    
# Setup two smooth walls implemented as 'planar interactions'
if include_walls:
    wall_potential = gp.apply_shifted_force_cutoff(gp.make_LJ_m_n(9,3))
    A = 4.0*math.pi/3*rho
    potential_params_list = [[A/15.0, -A/2.0, 3.0], [A/15.0, -A/2.0, 3.0]]    # Ingebrigtsen & Dyre (2014)
    particles_list =        [np.arange(c1.N),       np.arange(c1.N)]          # All particles feel the walls
    wall_point_list =       [[0, 0, wall_dist/2.0], [0, 0, -wall_dist/2.0] ]
    normal_vector_list =    [[0,0,1],               [0,0,-1]]                 # Carefull!
    walls = gp.setup_planar_interactions(c1, wall_potential, potential_params_list,
                                        particles_list, wall_point_list, normal_vector_list, compute_plan, verbose=True)

# Add gravity. NOTE: Carefull about PBC, since planar interactions takes abs(distance)
if include_gravity:
    potential = numba.njit(gp.make_IPL_n(-1)) # numba.njit should not be necesarry
    mg = 2
    potential_params_list = [[mg, 10*wall_dist],]       # Big cutoff, to avoid weird PBC effects
    particles_list =        [np.arange(c1.N),]          # All particles feel the gravity
    point_list =            [[0, 0, -wall_dist/2.0] ]   # Defining 0 for potential energy
    normal_vector_list =    [[0,0,1],     ]
    gravity = gp.setup_planar_interactions(c1, potential, potential_params_list,
                                        particles_list, point_list, normal_vector_list, compute_plan, verbose=True)
    
# Other features you can setup with planar interactions, using different potential-functions include:
# - External electrical field 
# - Semipermeable membranes (using energy-barriers)
# - Semi-2D simulations (eg., harmonic potential in the z-direction)

exclusions = None
if include_springs:
    exclusions = bonds['exclusions'] # Should be a list, which could be empty

# Setup pair interactions
pair_potential = gp.apply_shifted_force_cutoff(gp.make_LJ_m_n(12,6))
sigma =   [[1.0, 0.88], [0.88, 0.80]] # Setting up KABLJ. If all particles are type 0, 
epsilon = [[1.0, 0.50], [0.50, 1.50]] # ... this reverts to single componant LJ
cutoff = np.array(sigma)*2.5
params = gp.LJ_12_6_params_from_sigma_epsilon_cutoff(sigma, epsilon, cutoff)
LJ = gp.PairPotential(c1, pair_potential, params=params, max_num_nbs=1000, compute_plan=compute_plan)
pairs = LJ.get_interactions(c1, exclusions=exclusions, compute_plan=compute_plan, verbose=True)

# Add up interactions (For now: pair_interaction needs to be first, and there can be only one)
interactions_list = [pairs,]
if include_springs:
    interactions_list.append(bonds)
if include_walls:
    interactions_list.append(walls)
if include_gravity:
    interactions_list.append(gravity)

interactions, interaction_params = gp.add_interactions_list(c1, interactions_list, compute_plan, verbose=True,)

T0 = gp.make_function_ramp(value0=10.0, x0=10.0, value1=1.8, x1=20.0)
#T1 = gp.make_function_constant(value= 1.8)
T1 = gp.make_function_ramp(value0=1.8, x0=200., value1=1.2, x1=400)
#T1 = gp.make_function_sin(offset=0.45, period=200, amplitude=0.1)

# Setup NVT intergrator(s)
integrate0, integrator_params0 = nvt_nh.setup(c1, interactions, T0, tau=0.2, dt=0.001, compute_plan=compute_plan) # Equilibrate

dt = 0.0025
integrate1,  integrator_params1 = nvt_nh.setup(c1, interactions, T1, tau=0.2, dt=dt, compute_plan=compute_plan) # Production

scalars_t = []
coordinates_t = []
tt = []

equil_steps = 30000
inner_steps = 1000
steps = 500

start = cuda.event()
end = cuda.event()

dr = np.zeros(3)
dz = np.array((0., 0., 1.))

@numba.njit()
def get_bond_lengths_theta_z(r, bond_indices, dist_sq_dr_function, simbox_data):
    bond_lengths = np.zeros(bond_indices.shape[0], dtype=np.float32)
    theta_z = np.zeros(bond_indices.shape[0])
    dr = np.zeros(3)

    for j in range(bond_indices.shape[0]):
        dist_sq = dist_sq_dr_function(r[bond_indices[j,0]], r[bond_indices[j,1]], simbox_data, dr)
        dist = math.sqrt(dist_sq)
        bond_lengths[j] = dist
        theta_z[j] = math.acos(abs(dr[2]/dist))/math.pi*180
    return bond_lengths, theta_z


#Equilibration
zero = np.float32(0.0)
integrate0(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, interaction_params, integrator_params0, zero, equil_steps)
bond_lengths = []
theta_z = []

f = numba.njit(c1.simbox.dist_sq_dr_function)

start.record()
for i in range(steps):
    time_zero = np.float32(i*inner_steps*dt)
    integrate1(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, interaction_params, integrator_params0, time_zero, inner_steps)
    scalars_t.append(np.sum(c1.d_scalars.copy_to_host(), axis=0))
    tt.append(i*inner_steps*dt)

    c1.copy_to_host()
    coordinates_t.append(c1['r'][:,wall_dimension])
    if include_springs:
        for bond_particles in bond_particles_list:
            lengths, theta = get_bond_lengths_theta_z(c1['r'], bond_particles, f, c1.simbox.get_lengths())
            bond_lengths.append(lengths)
            theta_z.append(theta)

end.record()
end.synchronize()
timing_numba = cuda.event_elapsed_time(start, end)
nbflag = LJ.nblist.d_nbflag.copy_to_host()    
tps = steps*inner_steps/timing_numba*1000

print('\tsteps :', steps*inner_steps)
print('\tnbflag : ', nbflag)
print('\ttime :', timing_numba/1000, 's')
print('\tTPS : ', tps )
   
df = pd.DataFrame(np.array(scalars_t), columns=c1.sid.keys())
df['t'] = np.array(tt)
df['Ttarget'] = numba.vectorize(T1)(np.array(tt))
df['vol'] = np.prod(c1.simbox.get_lengths())
if include_walls:
    df['vol'] /= c1.simbox.get_lengths()[wall_dimension] * wall_dist
   
gp.plot_scalars(df, c1.N, c1.D, figsize=(10,8), block=False)

if include_springs:
    plt.figure() 
    plt.hist(np.array(bond_lengths).flatten(), bins=100, density=True)
    plt.xlabel('bond length')
    plt.ylabel('p(bond length)')
    plt.show(block=False)
    
    # what is the distribution of theta_z for random directions
    dr = np.random.randn(steps*c1.N, c1.D)
    dl = np.sum(dr*dr, axis=1)**0.5
    dr /= np.tile(dl,(3,1)).T
    theta_z_random = np.arccos(np.abs(dr[:,0]))/math.pi*180
    
    plt.figure() 
    bins = 100
    hist, bin_edges = np.histogram(theta_z_random, bins=bins, range=(0, 90), density=True)
    dx = bin_edges[1] - bin_edges[0]
    x = bin_edges[:-1]+dx/2 
    plt.plot(x,hist, label='Random')

    hist, bin_edges = np.histogram(np.array(theta_z).flatten(), bins=bins, range=(0, 90), density=True)
    plt.plot(x,hist, label='Simulation')
    plt.xlabel('Theta (angle with z-axis)')
    plt.ylabel('p(Theta)')
    plt.legend()
    plt.show(block=False)

plt.figure()
plt.plot(df['Ttarget'], df['U'])
plt.xlabel('Ttarget')
plt.xlabel('Potential energy')
plt.show(block=False)

plt.figure()    
bins = 300
subset = np.array(coordinates_t)[:,c1.ptype==0].flatten()
hist, bin_edges = np.histogram(subset, bins=bins, range=(-wall_dist/2, +wall_dist/2))
dx = bin_edges[1] - bin_edges[0]
x = bin_edges[:-1]+dx/2 
y = hist/len(coordinates_t)/Lxy**2/dx
plt.plot(x, y, 'b-', label='A')
rhoA = np.sum(c1.ptype==0)/Lxy/Lxy/wall_dist
plt.plot(x, np.ones_like(x)*rhoA, 'b--', label=f'rhoA={rhoA:.3}')

rhoB = np.sum(c1.ptype==1)/Lxy/Lxy/wall_dist
if rhoB>0: # We got some B-particles
    subset = np.array(coordinates_t)[:,c1.ptype==1].flatten()
    hist, bin_edges = np.histogram(subset, bins=bins, range=(-wall_dist/2, +wall_dist/2))
    y = hist/len(coordinates_t)/Lxy**2/dx
    plt.plot(x, y, 'g-', label='B')
    plt.plot(x, np.ones_like(x)*rhoB, 'g--', label=f'rhoB={rhoB:.3}')
    plt.plot(x, y*rhoA/rhoB, 'r--', label='B*(rhoA/rhoB)')

plt.xlabel('z')
plt.ylabel('rho(z)')
plt.legend()
plt.show()

