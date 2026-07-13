""" Test that the s variable returned by LJ pair potential (both the shifted potential and shifted force cutoffs), is 
consistent with the gradient of the energy variable.

Could extend to include other pair potentials but need to have at least a rough estimate of the third derivative to calculate
the error estimate
"""


def test_energy_gradient():
    import gamdpy as gp
    import numpy as np

    sig, eps, cut = 1.0, 1.0, 2.5
    r_min = 0.8
    h = 0.001

    pair_func_SP = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
    pair_pot_SP = gp.PairPotential(pair_func_SP, params=[sig, eps, cut], max_num_nbs=10)

    pair_func_SF = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
    pair_pot_SF = gp.PairPotential(pair_func_SF, params=[sig, eps, cut], max_num_nbs=10)

    cut_inner = 2.0
    pair_func_CS = gp.apply_cubic_spline_cutoff(gp.LJ_12_6_sigma_epsilon)
    pair_pot_CS = gp.PairPotential(pair_func_CS, params=[sig, eps, cut_inner, cut], max_num_nbs=10)

    r_values = np.arange(r_min, cut, h)
    v_values_SP = np.zeros_like(r_values)
    s_values_SP = np.zeros_like(r_values)
    v_values_SF = np.zeros_like(r_values)
    s_values_SF = np.zeros_like(r_values)
    v_values_CS = np.zeros_like(r_values)
    s_values_CS = np.zeros_like(r_values)


    params_SP, max_cut = pair_pot_SP.convert_user_params()
    params_SF, max_cut = pair_pot_SF.convert_user_params()
    params_CS, max_cut = pair_pot_CS.convert_user_params()
    for idx in range(len(r_values)):
        v_values_SP[idx], s_values_SP[idx], _ = pair_pot_SP.pairpotential_function(r_values[idx], params_SP[0,0])
        v_values_SF[idx], s_values_SF[idx], _ = pair_pot_SF.pairpotential_function(r_values[idx], params_SF[0,0])
        v_values_CS[idx], s_values_CS[idx], _ = pair_pot_CS.pairpotential_function(r_values[idx], params_CS[0,0])



    gradient_SP = np.gradient(v_values_SP, r_values)
    gradient_SF = np.gradient(v_values_SF, r_values)
    gradient_CS = np.gradient(v_values_CS, r_values)


    grad_difference_SP = gradient_SP + s_values_SP*r_values
    grad_difference_SF = gradient_SF + s_values_SF*r_values
    grad_difference_CS = gradient_CS + s_values_CS*r_values

    norm_SP = np.sqrt( np.sum(grad_difference_SP[1:]**2) )
    norm_SF = np.sqrt( np.sum(grad_difference_SF[1:]**2) )
    norm_CS = np.sqrt( np.sum(grad_difference_CS[1:]**2) )

    rss_v3 = 1.226e6 # root sum of squares of third derivative; determined analytically from the unshifted LJ potential and
    # summed over the relevant range and spacing of r-values . This is not affected by shifted potential/force cutoff
    # methods. For cubic-spline cutoff it differs in the spline region but this is a small difference compared to the contribution from the rest of the range
    error_estimate = rss_v3 * h**2/6

    # observed norm is 0.2044; error estimate is 0.2043. Include a safety factor of two.
    assert norm_SP < 2*error_estimate # I get 0.2044 for the shifted potential
    assert norm_SF < 2*error_estimate # I get 0.0418 for the corrected shifted force
    assert norm_CS < 2*error_estimate
    print(norm_SP)
    print(norm_SF)
    print(norm_CS)


if __name__ == "__main__":
    test_energy_gradient()