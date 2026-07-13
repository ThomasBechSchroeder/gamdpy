from pprint import pprint

import pytest
import scipy.constants as const

import gamdpy as gp


def test_conversion_factors_against_scipy_constants():
    """ Validate that conversion_factors matches SciPy constants for a set of known conversions and derived units. """

    cf = gp.conversion_factors(unit_length_in_Angstrom=1.0, unit_energy=1.0, unit_mass=1.0)
    assert cf["unit_length"] == pytest.approx(1e-10)
    assert cf["m"] == pytest.approx(1e-10)
    assert cf["Å"] == pytest.approx(1.0)
    assert cf["nm"] == pytest.approx(0.1)

    cf = gp.conversion_factors(unit_length=1.0, unit_energy_in_K=1.0, unit_mass=1.0)
    assert cf["unit_energy"] == pytest.approx(const.Boltzmann)
    assert cf["K"] == pytest.approx(1.0)

    cf = gp.conversion_factors(unit_length=1.0, unit_energy_in_kJ_per_mol=1.0, unit_mass=1.0)
    assert cf["unit_energy"] == pytest.approx(1e3 / const.Avogadro)

    cf = gp.conversion_factors(unit_length=1.0, unit_energy_in_kcal_per_mol=1.0, unit_mass=1.0)
    assert cf["unit_energy"] == pytest.approx(const.calorie / const.Avogadro)

    cf = gp.conversion_factors(unit_length=1.0, unit_energy=1.0, unit_mass=1.0)
    assert cf["kJ/mol"] == pytest.approx(const.Avogadro * 1e-3)

    cf = gp.conversion_factors(unit_length=1.0, unit_energy=1.0, unit_mass_in_u=1.0)
    assert cf["unit_mass"] == pytest.approx(const.atomic_mass)

    cf = gp.conversion_factors(unit_length=1.0, unit_energy=1.0, unit_mass_in_gram_per_mol=1.0)
    assert cf["unit_mass"] == pytest.approx(1e-3 / const.Avogadro)

    # Water-like units: L = 1 nm, E = k_B * 300 K, m = 18 g/mol
    L = 1.0  # nm
    T = 300.0  # K
    m_gmol = 18.0  # g/mol

    cf = gp.conversion_factors(
        unit_length_in_nm=L,
        unit_energy_in_K=T,
        unit_mass_in_gram_per_mol=m_gmol,
    )

    unit_length_SI = L * 1e-9
    unit_energy_SI = const.Boltzmann * T
    unit_mass_SI = (m_gmol * 1e-3) / const.Avogadro

    unit_time_ref = unit_length_SI * (unit_mass_SI / unit_energy_SI) ** 0.5
    unit_pressure_ref = unit_energy_SI / unit_length_SI**3
    unit_density_ref = unit_mass_SI / unit_length_SI**3

    assert cf["unit_length"] == pytest.approx(unit_length_SI)
    assert cf["unit_energy"] == pytest.approx(unit_energy_SI)
    assert cf["unit_mass"] == pytest.approx(unit_mass_SI)

    assert cf["unit_time"] == pytest.approx(unit_time_ref)
    assert cf["unit_pressure"] == pytest.approx(unit_pressure_ref)
    assert cf["unit_density"] == pytest.approx(unit_density_ref)

    cf = gp.conversion_factors(unit_length=1.0, unit_energy=1.0, unit_mass=1.0)
    assert cf["atm"] == pytest.approx(1.0 / const.atm)
    assert cf["psi"] == pytest.approx(1.0 / const.psi)

    cf_ag = gp.conversion_factors(unit_length=1.0, unit_energy=1.0, unit_mass_in_attograms=1.0)
    assert cf_ag["unit_mass"] == pytest.approx(1e-21)
    assert cf_ag["ag"] == pytest.approx(1.0)


def test_conversion_factors(verbose=False):

    # Test that wrong input raises errors
    with pytest.raises(KeyError) as e:
        gp.conversion_factors(unit_energy=1.0, unit_energy_in_K=120)
        assert "Expected only one unit_energy" in str(e.value)
    with pytest.raises(KeyError) as e:
        gp.conversion_factors(unit_length=1.0, unit_length_in_nm=3.4)
        assert "Expected only one unit_length" in str(e.value)

    if verbose:
        print("  ..:: Possible keyword inputs ::.. ")
        pprint(gp.conversion_factors(get_possible_inputs=True))

    # Test default behaviour
    cf_si = gp.conversion_factors(unit_length=1.0, unit_energy=1.0, unit_mass=1.0)  # standard SI units
    cf_si2 = gp.conversion_factors()
    assert cf_si == cf_si2, "The two methods for SI units are not equal"

    if verbose:
        print("  ..:: Standard SI units ::.. ")
        pprint(cf_si)

    # Test Argon units
    cf_Argon = gp.conversion_factors(unit_length_in_Angstrom=3.4, unit_energy_in_K=120.0, unit_mass_in_u = 39.948)
    cf_Argon2 = gp.conversion_factors(unit_length = 3.4e-10, unit_energy_in_kJ_per_mol=0.9978, unit_mass = 39.948*1.6605e-27)
    cf_Argon3 = gp.conversion_factors(unit_length_in_nm = 0.34, unit_energy_in_kcal_per_mol=0.2385, unit_mass_in_gram_per_mol = 39.948)
    for key, val in cf_Argon.items():
        assert cf_Argon2[key] == pytest.approx(val, rel=1e-3), "Conversion factors of two Argon inputs are not equal (2)"
        assert cf_Argon3[key] == pytest.approx(val, rel=1e-3), "Conversion factors of two Argon inputs are not equal (3)"

    if verbose:
        print("  ..:: Argon units ::.. ")
        pprint(cf_Argon)

    # Test cf_cgs units
    cf_cgs = gp.conversion_factors(unit_length_in_cm=1.0, unit_energy_in_erg=1.0, unit_mass_in_g=1.0)
    if verbose:
        print("  ..:: cgs units (centimetre–gram–second) ::.. ")
        pprint(cf_cgs)

    # Check some standar SI conversion factors
    cf = gp.conversion_factors()
    assert cf["(kcal/mol)/Angstrom"] == pytest.approx(cf["kcal/mol"] / cf["Angstrom"])

def test_conversion_factors_with_charge(verbose=False):
    cf = gp.conversion_factors(unit_length=1.0, unit_energy=1.0, unit_mass=1.0, unit_charge=1.0)
    assert cf["unit_charge"] == pytest.approx(1.0)
    assert 1.602176634e-19*cf["e"] == pytest.approx(1.0)
    assert 3.33564e-30*cf["D"] == pytest.approx(1.0)
    if verbose:
        print("  ..:: Conversion factors with charge ::.. ")
        pprint(cf)
    cf = gp.conversion_factors(unit_length=1.0, unit_energy=1.0, unit_mass=1.0, unit_charge_in_e=1.0)
    assert cf["e"] == pytest.approx(1.0)
    assert cf["unit_charge"] == pytest.approx(1.602176634e-19)
    assert cf["coulomb"] == pytest.approx(1.602176634e-19)
    if verbose:
        print("  ..:: Conversion factors with charge using elementary charge ::.. ")
        pprint(cf)

    # Dipole moment test. 1 e and 1 Å
    cf = gp.conversion_factors(
        unit_length_in_Angstrom=1.0,
        unit_energy_in_eV=1.0,
        unit_mass_in_u=1.0,
        unit_charge_in_e=1.0,
    )
    assert cf["Debye"] == pytest.approx(4.80320427)
    assert cf["coulomb_meter"] == pytest.approx(4.80320427*3.33564e-30)


if __name__ == "__main__":
    test_conversion_factors(verbose=True)
    test_conversion_factors_against_scipy_constants()
    test_conversion_factors_with_charge(verbose=True)
