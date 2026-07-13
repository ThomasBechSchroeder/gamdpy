import numpy as np
import pytest

import gamdpy as gp

def test_potential_functions_lennard_jones() -> None:
    # note: this example assumes these functions were implemented correctly in version bfa77f6e
    assert gp.LJ_12_6(1, [2, 3]) == (5.0, 42.0, 438.0), "Problem with gp.LJ_12_6"
    assert gp.LJ_12_6_sigma_epsilon(1, [2, 3]) == (48384.0, 585216.0, 7635456.0), "Problem with gp.LJ_12_6_sigma_epsilon"
    # gp.LJ_12_6_params_from_sigma_epsilon_cutoff seems not to be used
    #assert gp.LJ_12_6_params_from_sigma_epsilon_cutoff(1, [2, 3, 4]) == (5.0, 42.0, 438.0), "Problem with gp.LJ_12_6_params_from_sigma_epsilon_cutoff"
    # consider moving inner functions out for better testing

def test_make_potential_function_ipl_n() -> None:
    for n, r, a in [(12,1,1), (12,2**(1/6),3), (6,2**(1/6),1), (6,2,4), (1,2,3)]:
        ipl_n = gp.make_IPL_n(n)
        expected = (a*r**(-n), n*a*r**(-n-2), a*n*(n+1)*r**(-n-2))
        assert np.all(np.isclose(ipl_n(r, (a,)), expected)), f'Problem with make_IPL_n, {(n,r,a)=}'
    assert callable(gp.make_IPL_n(12)), "Problem with gp.make_IPL_n"

def test_add_potential_functions() -> None:
    LJ = gp.add_potential_functions(gp.make_IPL_n(12), gp.make_IPL_n(6, first_parameter=1))
    for r, a12, a6 in [(1,1,-1), (2**(1/6),3,-3), (2**(1/6),4,-4), (2,4,4), (2,4,-4)]:
        expected = gp.LJ_12_6(r, (a12, a6))
        assert np.all(np.isclose(LJ(r, (a12,a6)), expected, atol=1e-5)), f'Problem with  add_potential_functions, {(n,a12,a6)=}'

def test_bond_function_harmonic() -> None:
    assert gp.harmonic_bond_function(2.5, [2, 100]) == (12.5, -20.0, 100.0), "Problem with gp.harmonic_bond_function"
    # seems correct way: https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-variable-is-a-function

def test_make_potential_function_from_sympy() -> None:
    from sympy.abc import r,s,e
    potLJ = 4*e*((s/r)**(12)-(s/r)**6)
    potLJ_gp = gp.make_potential_function_from_sympy(potLJ, (s, e))
    assert potLJ_gp(1, (2,3)) == gp.LJ_12_6_sigma_epsilon(1, [2, 3]), "Problem with gp.make_potential_function_from_sympy"


def test_gaussian_core_model():
    sigma = 2.34
    epsilon = 3.21
    params = np.array([sigma, epsilon])
    r = 0.987
    kappa = 1 / sigma**2

    u_expected = epsilon * np.exp(-kappa * r * r)
    s_expected = u_expected * (2 * kappa)
    d2u_expected = u_expected * (4 * kappa**2 * r**2 - 2 * kappa)
    u, s, d2u = gp.gaussian_core_model(r, params)
    assert np.isclose(u, u_expected)
    assert np.isclose(s, s_expected)
    assert np.isclose(d2u, d2u_expected)


def test_exponential_repulsion() -> None:
    sigma, epsilon = params = 1.234, 0.987
    dist = 0.456
    pot_exp = gp.exponential_repulsion(dist, params)
    assert len(pot_exp) == 3
    u_expected = epsilon * np.exp( -dist / sigma )
    assert pot_exp[0] == pytest.approx(u_expected)
    s_expected = epsilon * np.exp( -dist / sigma ) / sigma / dist
    assert pot_exp[1] == pytest.approx(s_expected)
    du2dr2 = epsilon * np.exp( -dist / sigma ) / (sigma**2)
    assert pot_exp[2] == pytest.approx(du2dr2)

def test_potential_function_yukawa() -> None:
    params = [1.23, 0.891]
    dist = 3.45
    pot_yukawa = gp.yukawa(dist, params)
    assert len(pot_yukawa) == 3, "Problem with gp.yukawa"
    # print(pot_yukawa)
    u_yukawa_test = params[1]*params[0]*np.exp(-dist/params[0])/dist
    # print(u_yukawa_test, pot_yukawa)
    assert np.isclose(pot_yukawa[0], u_yukawa_test), "Problem with gp.yukawa"
    assert np.isclose(pot_yukawa[1], 0.0061450177), "Problem with gp.yukawa"
    assert np.isclose(pot_yukawa[2], 0.02499608), "Problem with gp.yukawa"

def test_potential_function_zbl() -> None:
    sigma, epsilon = params = 1.23, 0.92
    r = 1.364
    pot_zbl = gp.universal_zbl_potential(r, params)
    # print(pot_zbl)
    cs = 0.18175, 0.50986, 0.28022, 0.02817
    bs = 3.19980, 0.94229, 0.40290, 0.20162
    u_check = 0.0
    for i in range(4):
        u_check += epsilon * cs[i] * sigma * np.exp(-bs[i] * r / sigma) / r
    assert np.isclose(pot_zbl[0], u_check), "Problem with gp.universal_zbl_potential"
    assert np.isclose(pot_zbl[1], 0.3020571)
    assert np.isclose(pot_zbl[2], 0.73724324)

def test_potential_function_harmonic_repulsion() -> None:
    # u=(1-r)²
    params = 2.0, 1.0
    pair_pot = gp.PairPotential(gp.harmonic_repulsion, params=params, max_num_nbs=128)
    dist = 0.5
    pot_harm_rep = gp.harmonic_repulsion(dist, params)
    assert np.isclose(pot_harm_rep[0],0.25), "Problem with gp.harmonic_repulsion"
    assert np.isclose(pot_harm_rep[1],2.0), "Problem with gp.harmonic_repulsion"
    assert np.isclose(pot_harm_rep[2],2.0), "Problem with gp.harmonic_repulsion"
    eps, sig = 1.43, 1.37
    r = 0.98
    pot_harm_rep_2 = gp.harmonic_repulsion(r, [eps, sig])
    assert np.isclose(pot_harm_rep_2[0], np.float32(0.5*eps*(1.0-r/sig)**2)), f"Problem with gp.harmonic_repulsion"
    du_dr = -eps*(1.0-r/sig)/sig
    assert np.isclose(pot_harm_rep_2[1], -du_dr/r), "Problem with gp.harmonic_repulsion"
    assert np.isclose(pot_harm_rep_2[2], eps/sig**2), "Problem with gp.harmonic_repulsion"

def test_potential_function_hertzian() -> None:
    # u=eps*(1-r/sig)**alpha
    params = 1.0, 2.0, 1.0  # Same as "harmonic repulsion" above
    dist = 0.5
    pot_hertzian = gp.hertzian(dist, params)
    assert np.isclose(pot_hertzian[0],0.25), "Problem with gp.hertzian"
    assert np.isclose(pot_hertzian[1],2.0), "Problem with gp.hertzian"
    assert np.isclose(pot_hertzian[2],2.0), "Problem with gp.hertzian"
    eps, alpha, sig = 1.43, 3.1, 1.24
    r = 0.98
    pot_hertzian_2 = gp.hertzian(r, [eps, alpha, sig])
    assert np.isclose(pot_hertzian_2[0] , eps*(1.0-r/sig)**alpha ), "Problem with gp.hertzian"
    assert np.isclose(pot_hertzian_2[1] , alpha*eps*(1.0-r/sig)**(alpha-1)/sig/r ), "Problem with gp.hertzian"
    assert np.isclose(pot_hertzian_2[2] , eps*alpha*(alpha-1)*(1.0-r/sig)**(alpha-2)/sig/sig ), "Problem with gp.hertzian"

    # needs to add test for apply_shifted_force_cutoff, apply_shifted_potential_cutoff

def test_potential_function_saap() -> None:
    number_of_params = 8
    params = [1.0]*number_of_params
    dist = 1.0
    pot_SAAP = gp.SAAP(dist, params)
    assert len(pot_SAAP) == 3, "Problem with gp.SAAP"

def test_potential_function_saap_params() -> None:
    prm = gp.interactions.potential_parameters.SAAP_Deiters2019_params
    expected_keys={'comment', 'reference', 'units', 'Ne', 'Ar', 'Kr', 'Xe'}
    for key in expected_keys:
        assert key in prm.keys(), f"Problem with SAAP_Deiters2019_params, missing key: {key}"
    for element in 'Ne', 'Ar', 'Kr', 'Xe':
        this = prm[element]
        expected_keys = {'name', 'Z', 'M', 'eps', 'sig', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5'}
        for key in expected_keys:
            assert key in this.keys(), f"Problem with SAAP_Deiters2019_params['{element}'] missing key: {key}"

if __name__ == '__main__':  # pragma: no cover
    test_potential_functions_lennard_jones()
    test_add_potential_functions()
    test_bond_function_harmonic()
    test_make_potential_function_ipl_n()
    test_make_potential_function_from_sympy()
    test_gaussian_core_model()
    test_exponential_repulsion()
    test_potential_function_yukawa()
    test_potential_function_zbl()
    test_potential_function_harmonic_repulsion()
    test_potential_function_hertzian()
    test_potential_function_saap()
    test_potential_function_saap_params()
