from math import exp

import numba

def universal_zbl_potential(dist, params):
    r""" The Ziegler-Biersack-Littmark (ZBL) universal potential for high-energy pair collisions of atoms.

    The ZBL potential is a sum of four screened Coulomb interactions [Ziegler1985]_ :

    .. math::

        u(r) = \varepsilon \sum^4_{k=1} \frac{\sigma}{r} c_k\exp(-b_k r/\sigma)

    where the c's are [0.18175, 0.50986, 0.28022, 0.02817] and the b's are [3.19980, 0.94229, 0.40290, 0.20162].
    These are the same values as used in the LAMMPS implementation.

    Consider to atoms with atomic numbers :math:`Z_i` and :math:`Z_j`. Then the screening length is

    .. math::

        \sigma = \frac{0.46850 \text{Å}}{Z_i^{0.23}+Z_j^{0.23}}

    and the energy parameter is

    .. math::

        \varepsilon = \frac{Z_i Z_j e^2}{4 \pi \epsilon_0 \sigma}

    As an example, for copper (:math:`Z_i=Z_j=29`) the screening length is :math:`\sigma=0.1080` Å,
    the energy parameter is :math:`\varepsilon=1.122\times10^5` eV, :math:`\varepsilon/k_B=1.301\times10^9` K.

    Parameters
    ----------

    dist : float
        Distance between particles

    params : array-like
        :math:`\sigma`, :math:`\varepsilon`


    References
    ----------

    .. [Ziegler1985] JF Ziegler, JP Biersack, U Littmark "The Stopping and Range of Ions in Solids", Pergamon, 1985

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
