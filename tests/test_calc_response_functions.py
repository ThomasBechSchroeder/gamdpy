import gamdpy as gp
import numpy as np
import pytest

def test_calculate_response_functions_NpT():

    np.random.seed(2026)  # Set seed for reproducibility
    n = 10_000  # Number of datapoints in synthetic data

    out_arg = gp.tools.thermodynamics_NpT(
        1254,
        1251,
        2449+np.random.randn(n)*4.1,
        1244+np.random.randn(n)*2.2,
        3674+np.random.randn(n)*6.5,
        1000+np.random.randn(n)*8.2,
        1.23,
        0.876,
        3.432,
        False
    )

    out = gp.tools.thermodynamics_NpT(
        N=1254,
        dof=1251,
        U=2449+np.random.randn(n)*4.1,
        W=1244+np.random.randn(n)*2.2,
        K=3674+np.random.randn(n)*6.5,
        Vol=1000+np.random.randn(n)*8.2,
        k_B=1.23,
        T_ext=0.876,
        p_ext=3.432,
        per_particle=False
    )
    assert type(out) == dict

    #print(out)
    ## Generated with above
    expected = {'density': 1.2542044778216794, 'specific_volume': 0.7973713880424889,
     'jensen_bias_density_volume': 6.676536977745506e-05, 'potential_energy': 1.9529325489435667,
     'kinetic_energy': 2.9298631269371653, 'internal_energy': 4.8827956758807325, 'external_temperature': 0.876,
     'kinetic_temperature': 4.775429557075258, 'external_pressure': 3.432, 'pressure': 8.61112915846658,
     'compressibility_factor': 2.539625707462983, 'enthalpy': 7.619374279642554,
     'isobaric_heat_capacity': 0.719717102182806, 'isothermal_compressibility': 0.061953206277420306,
     'isothermal_bulk_modulus': 16.141214637416816, 'isobaric_expansion_coefficient': 0.24365934735003494,
     'isochoric_heat_capacity': 0.05039044033372986, 'isochoric_heat_capacity_excess': -0.5631382678002415,
     'adiabatic_index': 14.282810338949329, 'adiabatic_compressibility': 0.004337606171838147,
     'adiabatic_bulk_modulus': 230.54190730649714, 'isochoric_pressure_coefficient': 3.9329578239898124,
     'isochoric_pressure_coefficient_excess': 2.3902863162691466, 'adiabatic_pressure_coefficient': 4.2290516267039235,
     'adiabatic_expansion_coefficient': -0.018343960437013113, 'thermodynamic_gruneisen_parameter': 62.23042812021032,
     'configurational_adiabatic_scaling_exponent': -3.384281950112405, 'joule_thomson_coefficient': -0.8713619702829014}

    for key in expected.keys():
        pytest.approx(expected[key]) == out[key], f"Key {key} not equal to expected values"

    for key in out.keys():
        assert key in out_arg.keys()
        pytest.approx(out_arg[key]) == out[key]


def sample_harmonic_oscillator(n_samples, k=1.0, m=1.0, T=1.0):
    # Note. k_B = 1.0
    sigma_x = np.sqrt(T / k)
    sigma_p = np.sqrt(m * T)

    x = np.random.normal(0, sigma_x, size=n_samples)
    p = np.random.normal(0, sigma_p, size=n_samples)

    U = 0.5 * k * x ** 2
    K = p ** 2 / (2 * m)
    W = k * x ** 2

    return U, W, K

def test_harmonic_oscillator_response_functions():
    k = 2.35  # spring constant
    T = 1.2342  # physical temperature
    m = 1.23  # mass of particle
    n = 2000000  # sample size

    U, W, K = sample_harmonic_oscillator(n, k=k, m=m, T=T)

    data = gp.tools.thermodynamics_NVT(
        N=1,
        dof=1,
        V=1.0,
        U=U,
        W=W,
        K=K,
        k_B=1.0,
        T_ext=T,
        per_particle=True
    )

    # Analytical values
    analytic = {
        "potential_energy": 0.5 * T,
        "kinetic_energy": 0.5 * T,
        "internal_energy": T,
        "isochoric_heat_capacity": 1.0,
        "isochoric_heat_capacity_excess": 0.5,
        "configurational_adiabatic_scaling_exponent": 2.0,
        "canonical_virial_energy_correlation": 1.0,
    }

    # Assert that the analytical values are close to the calculated values
    tol = 5e-3
    for key, expected in analytic.items():
        value = data[key]
        assert abs(value - expected) < tol, f"{key}: {value} vs {expected}"


if __name__ == "__main__":
    test_calculate_response_functions_NpT()
    test_harmonic_oscillator_response_functions()
