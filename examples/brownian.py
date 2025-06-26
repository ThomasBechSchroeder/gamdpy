""" Brownian dynamics  """
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from numba import njit
import gamdpy as gp

temperature = 0.01
density = 0.5

# Setup configuration (give temperature kick to particles to get closer to equilibrium)
configuration = gp.Configuration(D=2)
configuration.make_positions(N=512, rho=density)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=2 * temperature, seed=0)

# Setup pair potential.
pairfunc = njit(gp.harmonic_repulsion)
eps, sig = 1.0, 1.0
pairpot = gp.PairPotential(pairfunc, params=[eps, sig], max_num_nbs=1000)
interactios = [pairpot, ]

# Setup integrator
dt = 0.005
tau = 0.1
integrator = gp.integrators.Brownian(temperature=temperature, tau=tau, dt=dt, seed=2025)
runtime_actions = [
    gp.TrajectorySaver(),
    gp.MomentumReset(steps_between_reset=100),
    gp.ScalarSaver(16, {'W' :True, 'K' :True}),
]

sim = gp.Simulation(configuration, interactios, integrator, runtime_actions,
                    num_timeblocks=8,
                    steps_per_timeblock=1024,
                    storage='memory')

for _ in sim.run_timeblocks():
    print(sim.status(per_particle=True))
print(sim.summary())

# Plot particle positions
positions = sim.configuration['r']
X, Y = configuration.simbox.get_lengths()
fig, ax = plt.subplots(figsize=(5,5))
ax.set_title("Particle Positions in a Brownian Dynamics Simulation")
ax.set_aspect('equal')
for x, y in positions:
    c = Circle((x, y), radius=0.5, facecolor='white', edgecolor='black')
    ax.add_patch(c)
ax.set_xlim(-X/2, X/2)
ax.set_ylim(-Y/2, Y/2)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
