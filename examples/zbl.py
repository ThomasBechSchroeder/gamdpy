""" Simulation of Cu (Z=29) using universal Ziegler–Biersack–Littmark (ZBL) potential

The simulation is done in natural units of m, ε and σ of the ZBL potential.

"""

import numpy as np

import gamdpy as gp

def conversion_factors_zbl(Z=29, molar_mass=63.546):
    """ Return a dictionary with conversion factors from natural ZBL units to SI units """
    from scipy.constants import atomic_mass, elementary_charge, pi
    from scipy.constants import epsilon_0, electron_volt, Boltzmann

    unit_mass = molar_mass*atomic_mass  # Kilogram
    sigma = 0.46850/(Z**0.23+Z**0.23)  # Ångström
    unit_length = sigma*1e-10  # Meter

    unit_energy = (Z * elementary_charge)**2
    unit_energy /= 4 * pi * epsilon_0 * unit_length  #  Joule

    unit_temperature = unit_energy/Boltzmann  # Kelvin
    unit_time = (unit_mass/unit_energy)**(1/2)*unit_length  # Seconds
    unit_pressure = unit_energy/unit_length**3  # Pascal

    return dict(
        unit_mass = unit_mass,
        unit_length = unit_length,
        in_Angstrom=unit_length*1e10,
        unit_energy = unit_energy,
        in_eV = unit_energy/electron_volt,
        unit_pressure = unit_pressure,
        in_bar = unit_pressure*1e-5,
        in_GPa = unit_pressure*1e-9,
        unit_time = unit_time,
        in_ns = unit_time*1e9,
        in_ps = unit_time*1e12,
        unit_temperature = unit_temperature,
        in_K = unit_temperature
    )

cf = conversion_factors_zbl(29,63.546)

temperature_SI = 3500  # Kelvin
temperature = temperature_SI/cf['in_K']

fcc_lattice_constant = 4.2  # Ångström
atoms_per_unit_cell = 4
density = atoms_per_unit_cell/(fcc_lattice_constant/cf['in_Angstrom'])**3
rho_mass_SI = density * cf['unit_mass'] / cf['unit_length'] ** 3  # kg/m^3
rho_mass_g_cm3 = rho_mass_SI * 1e-3
print(f'Density: {rho_mass_g_cm3:.3f} g/cm^3')

# Simulation
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[10]*3, rho=density)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=temperature*2)

pair_func = gp.apply_gromacs_cutoff(gp.universal_zbl_potential)
sig, eps, cut_inner, cut_outer = 1.0, 1.0, 8.0/cf['in_Angstrom'], 10.0/cf['in_Angstrom']
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut_inner, cut_outer], max_num_nbs=1000)

tau_T = 0.1/cf['in_ps']
alpha = 1/tau_T
integrator = gp.integrators.NVT_Langevin(temperature=temperature, alpha=alpha, dt=0.004/cf['in_ps'], seed=2025)

runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(16),
                   gp.RestartSaver(),
                   gp.MomentumReset(100)]

sim = gp.Simulation(configuration, [pair_pot], integrator, runtime_actions,
                    num_timeblocks=4, steps_per_timeblock=1*1024,
                    storage='Data/zbl.h5')

for timeblock in sim.run_timeblocks():
    K = np.sum(sim.configuration['K'])
    U = np.sum(sim.configuration['U'])
    W = np.sum(sim.configuration['W'])

    N, D = configuration.N, configuration.D
    dof = D * N - D
    volume = configuration.get_volume()
    rho = N/volume
    T_kin = 2 * K / dof
    P = rho * T_kin + W / volume

    print(f'{timeblock:04}: U/N = {U * cf['in_eV'] / N:.3f} eV    p = {P * cf['in_GPa']:.2f} GPa    T = {T_kin * cf['in_K']:.0f} K')
