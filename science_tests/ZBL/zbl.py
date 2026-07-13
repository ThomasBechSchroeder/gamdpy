""" Cu (Z=29) using universal Ziegler–Biersack–Littmark (ZBL) potential

The simulation is done in natural units of m, ε and σ.

"""

import numpy as np

import gamdpy as gp

def conversion_factors_zbl(Z=29, molar_mass=63.546, verbose=False):
    """ Return a dictionary with conversion factors from natural ZBL units to SI units """
    from scipy.constants import atomic_mass, elementary_charge, pi
    from scipy.constants import epsilon_0, electron_volt, Boltzmann

    unit_mass = molar_mass*atomic_mass  # kg

    sigma = 0.46850/(Z**0.23+Z**0.23)  # Å
    if verbose:
        print(f'Unit length: σ = {sigma} Å')
    unit_length = sigma*1e-10  # m

    unit_energy = (Z * elementary_charge)**2
    unit_energy /= 4 * pi * epsilon_0 * unit_length  #  J
    if verbose:
        print(f'Unit energy : ε = {unit_energy/electron_volt:.3e} eV')

    unit_temperature = unit_energy/Boltzmann  # K
    if verbose:
        print(f'Unit temperature: T_ε = {unit_temperature:.3e} K')

    unit_time = (unit_mass/unit_energy)**(1/2)*unit_length
    if verbose:
        print(f'Unit time: {unit_time*1e12} ps')

    unit_pressure = unit_energy/unit_length**3
    if verbose:
        print(f'Unit pressure: p = {unit_pressure*1e9} Gpa')

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
        in_ps=unit_time*1e12,
        unit_temperature = unit_temperature,
        in_K = unit_temperature
    )

def main():
    cf = conversion_factors_zbl(29,63.546, True)

    temperature_SI = 3500  # K
    temperature = temperature_SI/cf['unit_temperature']

    fcc_lattice_constant = 4.2*1e-10  # m
    density = 4/(fcc_lattice_constant/cf['unit_length'])**3
    rho_mass_SI = density * cf['unit_mass'] / cf['unit_length'] ** 3  # kg/m^3
    rho_mass_g_cm3 = rho_mass_SI * 1e-3  # g/cm^3
    print(f'Density: {rho_mass_g_cm3} g/cm^3  (lammps: 5.6970417 g/cm^3)')

    # Simulation
    configuration = gp.Configuration(D=3)
    configuration.make_lattice(gp.unit_cells.FCC, cells=[10]*3, rho=density)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=temperature*2)

    pair_func = gp.apply_gromacs_cutoff(gp.universal_zbl_potential)
    sig, eps, cut_inner, cut_outer = 1.0, 1.0, 8.0e-10/cf['unit_length'], 10.0e-10/cf['unit_length']
    pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut_inner, cut_outer], max_num_nbs=1000)

    tau_T = 0.1/cf['in_ps']
    alpha = 1/tau_T
    # integrator = gp.integrators.NVT_Langevin(temperature=temperature, alpha=alpha, dt=0.004e-12/cf['unit_time'], seed=2025)
    integrator = gp.integrators.NVT(temperature=temperature, tau=tau_T, dt=0.004/cf['in_ps'])


    runtime_actions = [gp.TrajectorySaver(),
                       gp.ScalarSaver(16),
                       gp.RestartSaver(),
                       gp.MomentumReset(100)]

    sim = gp.Simulation(configuration, [pair_pot], integrator, runtime_actions,
                        num_timeblocks=32, steps_per_timeblock=1*1024,
                        storage='zbl.h5')

    for timeblock in sim.run_timeblocks():
        K = np.sum(sim.configuration['K'])
        U = np.sum(sim.configuration['U'])
        W = np.sum(sim.configuration['W'])

        N, D = configuration.N, configuration.D
        dof = D * N - D
        volume = configuration.get_volume()
        rho = N / volume
        T_kin = 2 * K / dof
        P = rho * T_kin + W / volume

        print(f'{timeblock:04}: U/N = {U * cf['in_eV'] / N:.3f} eV    p = {P * cf['in_GPa']:.2f} GPa    T = {T_kin * cf['in_K']:.0f} K')

        # Save a dump file
        # lmp_dump = gp.configuration_to_lammps(configuration=sim.configuration, timestep=sim.steps_per_block*timeblock)
        # print(lmp_dump, file=open('dump.lammps', 'a'))

if __name__ == '__main__':
    main()
