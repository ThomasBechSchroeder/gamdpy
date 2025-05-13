""" Example of a user defined potential, example of a Yukawa potential.

This script demonstrates how to define a user-defined potential function
and use it in a gamdpy simulation. The example uses the Yukawa potential
as an example, but the same approach can be used for other pair potentials.

Comments
--------

The pair potential function is JIT compiled using numba.cuda,
and it is often the time-consuming part of the simulation.
Thus, you may want to optimize this function. For example,
it is the experience that numba.float32 is faster than numba.float64.

For mathematical functions supported by numba.cuda use the math module,
as described in the numba documentation:

    https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html#math

This example uses a syntax similar to the backend of gamdpy, making it easy to
include the code in the package, and making it available to the community.

It is recommended ensuring that the analytical derivatives are correct.

"""

from math import exp  # Note math.exp is supported by numba cuda

import numpy as np
import matplotlib.pyplot as plt
import numba

import gamdpy as gp


def yukawa(dist, params):
    """ The Yukawa potential: u(r) = A·exp(-κ·r)/r

    parameters: κ, A    (κ is the greek letter kappa)

    The Yukawa potential is a simple screened Coulomb potential.
    The potential is given by:

        u(r) = A·exp(-κ·r)/r

    where A is the strength of the interaction,
    and kappa is the inverse of the screening length.

    The s(r) function, used to compute pair forces (𝐅=s·𝐫), is defined as

        s(r) = -u'(r)/r

    and specifically for the Yukawa potential it is

        s(r) = A·exp(-κ·r)·(κ·r + 1)/r³

    The second derivative (`d2u_dr2`) of the potential is given by

        u''(r) = A·exp(-κ·r)*([κ·r]² + 2κ·r + 2)/r³

    """

    # Extract parameters
    kappa = numba.float32(params[0])  # κ
    prefactor = numba.float32(params[1])  # A

    # Floats. Note: numba.float32's may make code faster
    one = numba.float32(1.0)
    two = numba.float32(2.0)

    # Compute helper variables
    kappa_dist = kappa * dist  # κ·r
    inv_dist = one / dist  # 1/r
    inv_dist3 = inv_dist*inv_dist*inv_dist  # 1/r³
    exp_kappa_dist = prefactor * exp(-kappa_dist)  # A·exp(-κ·r)

    # Compute pair potential energy, pair force and pair curvature

    # A·exp(-κ·r)/r
    u = exp_kappa_dist * inv_dist

    # A·exp(-κ·r)·(κ·r + 1)/r³
    s = exp_kappa_dist * (kappa_dist + one) * inv_dist3

    # A·exp(-κ·r)*([κ·r]² + 2κ·r + 2)/r³
    d2u_dr2 = exp_kappa_dist * (kappa_dist*kappa_dist + two * kappa_dist + two) * inv_dist3

    return u, s, d2u_dr2  # u(r), -u'(r)/r, u''(r)


# Plot the Yukawa potential, and confirm the analytical derivatives
# are as expected from the numerical derivatives.
plt.figure()
r = np.linspace(0.8, 3, 200, dtype=np.float32)
params = [1.0, 1.0, 2.5]
u = [yukawa(rr, params)[0] for rr in r]
u_check = params[1] * np.exp(-params[0] * r) / r
s = [yukawa(rr, params)[1] for rr in r]
s_numerical = -np.gradient(u, r) / r
umm = [yukawa(rr, params)[2] for rr in r]
umm_numerical = np.gradient(np.gradient(u, r), r)
plt.plot(r, u, '-', label='u(r)')
plt.plot(r, u_check, '--', label='u(r), check')
plt.plot(r, s, '-', label='s(r)')
plt.plot(r, s_numerical, '--', label='s(r), numerical')
plt.plot(r, umm, label='u\'\'(r)')
plt.plot(r, umm_numerical, '--', label='u\'\'(r), numerical')
plt.xlabel('r')
plt.ylabel('u, s, u\'\'')
plt.legend()
if __name__ == "__main__":
    plt.show()

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)

# Setup pair potential: Single component Yukawa system
pair_func = gp.apply_shifted_potential_cutoff(yukawa)  # Note: We use the above yukawa function here
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = gp.integrators.NVE(dt=0.005)

runtime_actions = [gp.MomentumReset(100),
                   gp.TrajectorySaver(),
                   gp.ScalarSaver(), ]

# Setup Simulation.
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=32, steps_per_timeblock=1024,
                    storage='memory')

# Run simulation
sim.run()
