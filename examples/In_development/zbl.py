"""
Implementation of the ZBL potential.

"""

from math import exp  # Note math.exp is supported by numba cuda

import numpy as np
import matplotlib.pyplot as plt
import numba

import gamdpy as gp


def my_zbl(dist, params):
    r""" The Ziegler-Biersack-Littmark (ZBL) universal potential for high-energy pair collisions of atoms.

    The ZBL potential is a sum of four screened Coulomb interactions:

    .. math::

        u(r) = \varepsilon \sum^4_{k=1} \frac{\sigma}{r} c_k\exp(-b_k r/\sigma)

    where c's are [0.18175, 0.50986, 0.28022, 0.02817] and b's are [3.19980, 0.94229, 0.40290, 0.20162].
    Consider to atoms with atomic numbers :math:`Z_i` and :math:`Z_j`. Then the screening length is

    .. math::

        \sigma = \frac{0.46850 \text{Å}}{Z_i^{0.23}+Z_j^{0.23}}

    and the energy parameter is

    .. math::

        \epsilon = \frac{Z_i Z_j e^2}{4 \pi \epsilon_0 \sigma}

    As an example, for Cobber (:math:`Z_i=Z_j=29`) the screening length is :math:`\sigma=0.1080` Å,
    the energy parameter is :math:`\epsilon=1.122\times10^5` eV (:math:`\epsilon/k_B=1.301\times10^9` K).

    Parameters
    ----------

    dist : float
        Distance between particles

    params : array-like
        :math:`\sigma`, :math:`\varepsilon`

    """

    # Use float32 for all calculations
    f32 = numba.float32
    zero = f32(0.0)
    one = f32(1.0)
    two = f32(2.0)

    # Extract parameters
    sigma = f32(params[0])
    epsilon = f32(params[1])

    # Universal paramters
    cs = f32(0.18175), f32(0.50986), f32(0.28022), f32(0.02817)
    bs = f32(3.19980), f32(0.94229), f32(0.40290), f32(0.20162)

    # Compute pair potential energy, pair force and pair curvature
    u = zero  # u = sum c·exp(-b*r)/r
    s = zero  # s = -u'(r)/r
    d2u_dr2 = zero  # d2u_dr2 = u''(r)

    dist_inv = one / dist
    dist_inv2 = dist_inv * dist_inv
    dist_inv3 = dist_inv2 * dist_inv
    sigma_inv = one / sigma

    for i in range(4):
        A = f32(epsilon * cs[i] * sigma)
        B = f32( bs[i] * sigma_inv )
        u += A * exp( - B * dist ) * dist_inv
        s += (B*dist_inv2 + dist_inv3) * A * exp( - B * dist)
        d2u_dr2 += (B*B*dist_inv + two*B*dist_inv2 + two*dist_inv3) * A * exp( - B * dist)

    return u, s, d2u_dr2


# Plot the potential and confirm the analytical derivatives
# are as expected from the numerical derivatives.
cs = 0.18175, 0.50986, 0.28022, 0.02817
bs = 3.19980, 0.94229, 0.40290, 0.20162

plt.figure()
r = np.linspace(0.1, 30, 1024*16, dtype=np.float32)
params = [1.00, 1.00, 30.0]
u_check = np.zeros_like(r)
for i in range(4):
    u_check += params[1] * cs[i] * params[0] * np.exp(-bs[i] * r / params[0]) / r
u = [gp.universal_zbl_potential(rr, params)[0] for rr in r]
s = [gp.universal_zbl_potential(rr, params)[1] for rr in r]
s_numerical = -np.gradient(u, r) / r
umm = [gp.universal_zbl_potential(rr, params)[2] for rr in r]
umm_numerical = np.gradient(np.gradient(u, r), r)
plt.plot(r, u, '-', label='u(r)')
plt.plot(r, u_check, '--', label='u(r), check')
plt.plot(r, s, '-', label='s(r)')
plt.plot(r, s_numerical, '--', label='s(r), numerical')
plt.plot(r, umm, label='u\'\'(r)')
plt.plot(r, umm_numerical, '--', label='u\'\'(r), numerical')
plt.xlim(1, 4)
plt.ylim(0, 1.5)
plt.xlabel('r')
plt.ylabel('u, s, u\'\'')
plt.legend()


# Reproducing fig 2-4 in "THE STOPPING AND RANGE OF IONS IN SOLIDS"
plt.figure()
plt.title('Reproducing fig 2-4 in "THE STOPPING AND RANGE OF IONS IN SOLIDS"')
plt.plot(r, u_check*r)
plt.yscale('log')
plt.ylim(1e-4, 1.0)
plt.xlim(0, 30)
plt.show()

if __name__ == "__main__":
    plt.show()

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.0001)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)

# Setup pair potential: Single component system
pair_func = gp.apply_shifted_potential_cutoff(gp.universal_zbl_potential)  # Note: We use the above yukawa function here
sig, eps, cut = 1.0, 1.0, 3.0
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=2000)

# Setup integrator: NVT
integrator = gp.integrators.NVE(dt=0.005)

runtime_actions = [gp.RestartSaver(),
                   gp.MomentumReset(100),
                   gp.TrajectorySaver(),
                   gp.ScalarSaver(16), ]

# Setup Simulation.
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=32, steps_per_timeblock=1024,
                    storage='Data/zbl.h5')

# Run simulation
for block in sim.run_timeblocks():
    print(f'{sim.status(per_particle=True)}')
print(sim.summary())
