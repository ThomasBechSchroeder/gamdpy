
import numba

def LJ_SF(dist, params):
    """ The 12-6 Lennard-Jones potential + Shifted force Coulomb potential

    See :func: examples/water.py for example

    dist : float
        Distance between particles

    params : array-like
        A_12, A_6, Q, cut-off

    Returns
    -------

    u : float
        Potential energy
    s : float
        Force multiplier, -u'(r)/r
    umm : float
        Second derivative of potential energy

    """

    sigma = params[0]  
    epsilon = params[1] 
    q = params[2]
    cutoff = params[3]

    one = numba.float32(1.0)
    OneOdist = one / dist  
    sigmaOdist = sigma * OneOdist

    u_vw = numba.float32(4.0) * epsilon * (sigmaOdist ** 12 - sigmaOdist ** 6)
    s_vw = numba.float32(24.0) * epsilon * (numba.float32(2.0) * sigmaOdist ** 12 - sigmaOdist ** 6) * OneOdist ** 2
    
    u_q = q*OneOdist
    s_q = q*OneOdist*(one/(dist**2) - one/(cutoff**2))

    d2u_vw = numba.float32(24.0) * epsilon * (numba.float32(26.0) * sigmaOdist ** 12 - numba.float32(7.0) * sigmaOdist ** 6) * OneOdist ** 2
    d2u_q = q*(one/(cutoff**2) + one/(dist**2))

    u = u_vw + u_q
    s = s_vw + s_q
    umm = d2u_vw + d2u_q

    return u, s, umm  


