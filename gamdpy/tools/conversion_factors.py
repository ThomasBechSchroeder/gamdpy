""" Function that return dictionary with conversion factors from reduced units to real units """
from scipy.constants import Boltzmann


def conversion_factors(**kwargs):
    """ Return conversion factors from simulation units to common physical units.

    The function defines a set of *base simulation units* by specifying what
    one unit of length, energy, and mass correspond to in SI (meters, joules,
    kilograms). From these three base units it derives such as time, pressure, and more.
    These are returned as a dictionary of multiplicative conversion factors.

    Interpretation of the returned dictionary
    -----------------------------------------
    Each entry `cf[key]` is a factor that converts a value expressed in
    simulation units to the unit indicated by `key`:
    Example: to give simulation time in ps then use `t_ps = t_sim * cf['ps']`.

    Parameters
    ----------
    **kwargs
        Exactly one keyword must be provided for each of the three base units:
        `unit_length*`, `unit_energy*`, and `unit_mass*`. The suffix determines
        the input unit, e.g.:

          - Length: `unit_length_in_Angstrom`, `unit_length_in_nm`, ...
          - Energy: `unit_energy_in_kJ_per_mol`, `unit_energy_in_K`, `unit_energy_in_eV`, ...
          - Mass:   `unit_mass_in_u`, `unit_mass_in_g`, ...

        If no keyword arguments are given, SI base units are assumed:
        `unit_length = 1 m`, `unit_energy = 1 J`, `unit_mass = 1 kg`.

        If the keyword provides a unit_charge, then the output dictionary contains information for charge conversions,
        and the value of Coulombs constant of the unit system.

        Special option:
        `get_possible_inputs=True` returns a dictionary mapping all accepted
        input keyword names to their conversion factors to SI.

    Returns
    -------
    dict
        Dictionary of conversion factors from simulation units to various units.
        The dictionary includes (among others):
          - length: m, cm, nm, Å, ...
          - energy: J, kJ/mol, kcal/mol, K, eV, ...
          - mass: kg, g, u, g/mol, ...
          - time: s, ps, fs, ns, ...
          - pressure: Pa, MPa, GPa, bar, atm, ...
          - ...

    Raises
    ------
    KeyError
        If more than one keyword is provided for length, energy, or mass.
    KeyError
        If any of length, energy, or mass is not specified (and kwargs are not empty).


    Examples
    --------

    >>> from pprint import pprint
    >>> from gamdpy.tools import conversion_factors
    >>> cf = conversion_factors(unit_length=1.0, unit_energy=1.0, unit_mass=1.0)  # Standard SI units
    >>> cf = conversion_factors()  # Standard SI units (same as above)
    >>> cf = conversion_factors(unit_length_in_Angstrom=3.4, unit_energy_in_K=120.0, unit_mass_in_u = 39.948) # Argon units
    >>> cf = conversion_factors(unit_length_in_Angstrom=1.0, unit_energy_in_kcal_per_mol=1.0, unit_mass_in_u=1.0)  # Molar units (LAMMPS real units)
    >>> cf = conversion_factors(unit_length_in_cm=1.0, unit_energy_in_erg=1.0, unit_mass_in_g=1.0)  # centimetre–gram–second (CGS) system
    >>> cf = conversion_factors(unit_length_in_nm=1.0, unit_energy_in_kJ_per_mol=1.0, unit_mass_in_u = 1.0)  # Atomistic SI-like units
    >>> cf = conversion_factors(unit_length_in_Angstrom=1.0, unit_energy_in_eV=1.0, unit_mass_in_u = 1.0)  # Metallic units
    >>> cf = conversion_factors(unit_length_in_Angstrom=3.165492, unit_energy_in_kcal_per_mol=0.1554253, unit_mass_in_u=15.999, unit_charge_in_e=1.0)  # SPC/Fw Oxygen Water units
    """
    from math import pi

    # Check that one, and only one energy, length and mass is given
    for prefix in 'unit_length', 'unit_energy', 'unit_mass':
        matches = [k for k in kwargs if k.startswith(prefix)]
        if len(matches) > 1:
            raise KeyError(f'Expected only one {prefix} key, but got {len(matches)}: {matches}')

    # Fundamental constants (same format as scipy.constants)
    Avogadro = 6.02214076e+23  # 1 / mol
    atomic_mass = 1.66053906892e-27  # kg
    Boltzmann = 1.380649e-23  # m² · kg / ( s² · K ) = J / K
    calorie = 4.184  # J
    electron_volt = 1.602176634e-19  # J
    astronomical_unit = 149597870700.0  # m
    parsec = 3.085677581491367e+16  # m
    atomic_mass = 1.66053906892e-27  # kg
    solar_masses = 1.988416e30  # kg
    hartree = 4.3597447222060e-18  # J
    bohr_radius = 5.29177210544e-11 # m
    elementary_charge = 1.602176634e-19  # C

    # What unit one in reduced units corresponds to in SI units
    unit_length, unit_energy, unit_mass = None, None, None
    if not kwargs:
        unit_length, unit_energy, unit_mass = 1.0, 1.0, 1.0  # Assume SI units of no kwargs are given

    possible_lengths = {
        # Name         # Conversion factor to SI
        'unit_length': 1.0,
        'unit_length_in_cm': 1e-2,
        'unit_length_in_nm': 1e-9,
        'unit_length_in_Å': 1e-10,
        'unit_length_in_A': 1e-10,
        'unit_length_in_Angstrom': 1e-10,
        'unit_length_in_Ångström': 1e-10,
        'unit_length_in_bohr_radius': bohr_radius,
        'unit_length_in_AU': astronomical_unit,
        'unit_length_in_parsec': parsec
    }
    for key, val in possible_lengths.items():
        if key in kwargs:
            unit_length = kwargs[key]*val

    possible_energies = {
        # Name         # Conversion factor to SI
        'unit_energy': 1.0,
        'unit_energy_in_K': Boltzmann,
        'unit_energy_in_Kelvin': Boltzmann,
        'unit_energy_in_kJ_per_mol': 1e3 / Avogadro,
        'unit_energy_in_kcal_per_mol': 1e3 * calorie / Avogadro,
        'unit_energy_in_erg': 1e-7,
        'unit_energy_in_zeptojoule': 1e-21,
        'unit_energy_in_hartree': hartree,
        'unit_energy_in_eV': electron_volt,
    }
    for key, val in possible_energies.items():
        if key in kwargs:
            unit_energy = kwargs[key]*val

    possible_masses = {
        # Name         # Conversion factor to SI
        'unit_mass': 1.0,
        'unit_mass_in_g': 1e-3,
        'unit_mass_in_u': atomic_mass,
        'unit_mass_in_amu': atomic_mass,
        'unit_mass_in_gram_per_mol': 1e-3 / Avogadro,
        'unit_mass_in_attograms': 1e-21
    }
    for key, val in possible_masses.items():
        if key in kwargs:
            unit_mass = kwargs[key]*val

    possible_charges = {
        'unit_charge': 1.0,  # Coulombs
        'unit_charge_in_Coulombs': 1.0,
        'unit_charge_in_elementary_charge': elementary_charge,
        'unit_charge_in_e': elementary_charge,
    }

    # Return a dict with possible inputs and their
    if 'get_possible_inputs' in kwargs:
        if kwargs['get_possible_inputs']:
            return possible_lengths | possible_energies | possible_masses | possible_charges

    # Check that a length, energy and mass have been given in keyword arguments
    if unit_length == None or unit_energy == None or unit_mass == None:
        print(f'{unit_length = }, {unit_energy = }, {unit_mass = }')
        print(f'{possible_lengths = }\n{possible_energies = }\n{possible_masses = }')
        raise KeyError(f'Some unit_length, unit_energy and unit_mass needs to be specified. Got {kwargs}.')

    # Derived units. N.B. Below we assume D=3 spatial dimensions
    unit_time = unit_length * (unit_mass / unit_energy) ** 0.5  # s
    unit_force = unit_energy / unit_length
    unit_pressure = unit_energy / unit_length ** 3  # Pa
    unit_density = unit_mass / unit_length ** 3  # kg/m^3
    unit_area = unit_length**2
    unit_volume = unit_length**3

    output = {
        # Length
        "unit_length": unit_length,  # m
        "in_parsec": unit_length / parsec,
        "AU": unit_length / astronomical_unit,
        "km": unit_length * 1e-3,
        "m": unit_length,
        "cm": unit_length * 1e2,
        "mm": unit_length * 1e3,
        "micrometer": unit_length * 1e6,
        "μm": unit_length * 1e6,
        "micron": unit_length * 1e6,
        "nm": unit_length * 1e9,
        "Å": unit_length * 1e10,
        "Angstrom": unit_length * 1e10,
        "Ångstrom": unit_length * 1e10,
        "Bohr_radius": unit_length / bohr_radius,

        # Energy
        "unit_energy": unit_energy,  # J
        "J": unit_energy,
        "zeptojoule": unit_energy*1e21,
        "zJ": unit_energy*1e21,
        "kJ": unit_energy * 1e-3,
        "kJ/mol": unit_energy * 1e-3 * Avogadro,
        "cal": unit_energy / calorie,
        "kcal": unit_energy * 1e-3 / calorie,
        "kcal/mol": unit_energy * 1e-3 / calorie * Avogadro,
        "K": unit_energy / Boltzmann, # K
        "Kelvin": unit_energy / Boltzmann,
        "electron_volt": unit_energy / electron_volt,
        "eV": unit_energy / electron_volt,
        "keV": unit_energy / electron_volt * 1e-3,
        "MeV": unit_energy / electron_volt * 1e-6,
        "GeV": unit_energy / electron_volt * 1e-9,
        "erg":  unit_energy * 1e7,
        "hartree": unit_energy / hartree,

        # Mass
        "unit_mass": unit_mass,  # kg
        "solar_masses": unit_mass/solar_masses,
        "M☉": unit_mass/solar_masses,
        "kg": unit_mass,
        "gram": unit_mass*1e3,
        "g": unit_mass*1e3,
        "mg": unit_mass*1e6,
        "mikrogram": unit_mass*1e9,
        "μg": unit_mass*1e9,
        "nanogram": unit_mass*1e12,
        "ng": unit_mass*1e12,
        "picogram": unit_mass*1e15,
        "pg": unit_mass*1e15,
        "attogram": unit_mass*1e21,
        "ag": unit_mass*1e21,
        "yoctogram": unit_mass*1e27,
        "yg": unit_mass*1e27,
        "u": unit_mass / atomic_mass,
        "amu": unit_mass / atomic_mass,
        "dalton": unit_mass / atomic_mass,
        "kg/mol": unit_mass * Avogadro,
        "g/mol": unit_mass * 1e3 * Avogadro,

        # Time
        'unit_time': unit_time,  # s
        'years': unit_time / 31556926,
        'weeks': unit_time / 604800,
        'days': unit_time / 86400,
        'hours': unit_time / 3600,
        'minutes': unit_time / 60,
        's': unit_time,
        'seconds': unit_time,
        'milliseconds': unit_time * 1e3,
        'ms': unit_time * 1e3,
        'microseconds': unit_time * 1e6,
        'μs': unit_time * 1e6,
        'nanoseconds': unit_time * 1e9,
        'ns': unit_time * 1e9,
        'picoseconds': unit_time * 1e12,
        'ps': unit_time * 1e12,
        'femtoseconds': unit_time * 1e15,
        'fs': unit_time * 1e15,

        # Force
        'unit_force': unit_force,  # N
        'Newton': unit_force,
        'N': unit_force,
        'millinewton': unit_force*1e3,
        'mN': unit_force*1e3,
        'nanonewton': unit_force*1e9,
        'nN': unit_force*1e9,
        'piconewton': unit_force*1e12,
        'pN': unit_force*1e12,
        'femtonewton': unit_force*1e15,
        'fN': unit_force*1e15,
        'Dyne': unit_force * 1e5,  # CGS system
        'dyn': unit_force * 1e5,
        '(kcal/mol)/Angstrom': unit_force * 1e-3 / calorie * Avogadro / 1e10,  # from LAMMPS real units
        '(kJ/mol)/Angstrom': unit_force * 1e-3 / Avogadro / 1e10,
        'eV/Angstrom': unit_force / electron_volt / 1e10,  # from LAMMPS metallic units

        # Pressure
        'unit_pressure': unit_pressure,
        'Pascal': unit_pressure,
        'pa': unit_pressure,
        'hectopascal': unit_pressure * 1e-2,
        'kilopascal': unit_pressure * 1e-3,
        'kpa': unit_pressure * 1e-3,
        'megapascal': unit_pressure * 1e-6,
        'MPa': unit_pressure * 1e-6,
        'gigapascal': unit_pressure * 1e-9,
        'GPa': unit_pressure * 1e-9,
        'bar': unit_pressure * 1e-5,
        'mbar': unit_pressure * 1e-2,
        'psi': unit_pressure / 6894.757293168361,
        'torr': unit_pressure / 133.322368,
        'atm': unit_pressure / 101325,
        'atmospheres': unit_pressure / 101325,
        'Barye': unit_pressure * 10,
        'Ba': unit_pressure * 10,

        # Density (3D)
        'unit_density': unit_density,  # kg/m³
        'kg/m3': unit_density,
        'g/ml': unit_density * 1e-3,
        'g/cm3': unit_density * 1e-3,
        'grams_per_cubic_centimeter': unit_density * 1e-3,
        'u/nm3': unit_density / atomic_mass * 1e-27,
        'u/Å3': unit_density / atomic_mass * 1e-30,

        # Area
        'unit_area': unit_area,  # m²
        'square_meter': unit_area,
        'm2': unit_area,
        'square_centimeter': unit_area*1e4,
        'cm2': unit_area*1e4,
        'square_millimeter': unit_area*1e6,
        'mm2': unit_area*1e6,
        'square_nanometer': unit_area*1e18,
        'nm2': unit_area*1e18,
        'square_Ångström': unit_area*1e20,
        'square_Angstrom': unit_area*1e20,
        'Å2': unit_area*1e20,

        # Volume
        'unit_volume': unit_volume,  # m³
        'cubic_meter': unit_volume,
        'm3': unit_volume,
        'cubic_centimeter': unit_volume*1e6,
        'cc': unit_volume*1e6,
        'cm3': unit_volume*1e6,
        'cubic_millimeter': unit_volume*1e9,
        'mm3': unit_volume*1e9,
        'cubic_nanometer': unit_volume*1e27,
        'nm3': unit_volume*1e27,
        'cubic_Angstrom': unit_volume*1e30,
        'cubic_Ångström': unit_volume*1e30,
        'Å3': unit_volume*1e30,

        # Angles
        'degrees': 180/pi,  # for converting from radians
    }

    # For handling charges and Columbs law
    unit_charge = None
    matches = [k for k in kwargs if k.startswith('unit_charge')]
    if len(matches) > 1:
        raise KeyError(f'Expected only one unit_charge key, but got {len(matches)}: {matches}')
    if len(matches) == 1:
        unit_charge = kwargs[matches[0]] * possible_charges[matches[0]]

    if unit_charge is not None:
        coulombs_constant_SI = 8987551786
        coulombs_constant = coulombs_constant_SI * unit_charge**2 / unit_force / unit_length ** 2
        charge_coulomb_natural_units = coulombs_constant ** 0.5  # Coulomb’s law becomes F = q1*q2 / r^2 in these units
        output.update({
            'unit_charge': unit_charge,
            'coulomb': unit_charge,
            'e': unit_charge/elementary_charge,
            'coulombs_constant': coulombs_constant,
            'k_e': coulombs_constant,
            'charge_coulomb_natural_units': charge_coulomb_natural_units,
            'coulomb_meter': unit_charge*unit_length,
            'D': unit_charge*unit_length/3.33564e-30,
            'Debye': unit_charge*unit_length/3.33564e-30
        })

    return output