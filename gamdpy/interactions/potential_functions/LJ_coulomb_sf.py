
import numba

def LJ_coulomb_sf(dist, params):
    """ The 12-6 Lennard-Jones shifted potential + Coulomb potential shifted force 

    See :func: examples/water.py for example

    dist : float
        Distance between particles

    params : array-like
        sigma, epsilon, charge, cut-off LJ, cut-off Coulomb

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
    cutoff_lj, cutoff_coulomb = params[3], params[4]
    
    one, two, four, twentyfour = numba.float32(1.0), numba.float32(2.0), numba.float32(4.0), numba.float32(24.0)

    OneOdist, OneOcutoff_lj = one/dist, one/cutoff_lj   
    sigmaOdist, sigmaOcutoff_lj = sigma * OneOdist, sigma*OneOcutoff_lj
    
    if dist < cutoff_lj: 
        u_lj = four*epsilon*(sigmaOdist**12 - sigmaOdist**6)
        u_lj_cut = four*epsilon*(sigmaOcutoff_lj**12 - sigmaOcutoff_lj**6) 
        u_lj = u_lj - u_lj_cut 

        s_lj = twentyfour*epsilon*(two*sigmaOdist**12 - sigmaOdist**6)*OneOdist**2
     
        d2u_lj = twentyfour*epsilon*(numba.float32(26.0)*sigmaOdist**12 - numba.float32(7.0)*sigmaOdist**6)*OneOdist**2
    
    if dist < cutoff_coulomb:
        u_q = q*OneOdist
        s_q = q*OneOdist*( one/(dist**2) - one/(cutoff_coulomb**2) )
        d2u_q = q*( one/(cutoff_coulomb**2) + one/(dist**2) )
   
    u = u_lj + u_q
    s = s_lj + s_q
    umm = d2u_lj + d2u_q

    return u, s, umm  
