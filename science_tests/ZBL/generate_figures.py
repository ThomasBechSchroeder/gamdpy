""" Analyse data from simulation """

def main():

    import matplotlib.pyplot as plt
    import numpy as np

    import gamdpy as gp

    from zbl import conversion_factors_zbl

    cf = conversion_factors_zbl(Z=29, molar_mass=63.546)
    print(cf)

    filename='./zbl.h5'
    output = gp.tools.TrajectoryIO(filename).get_h5()

    nblocks, nconfs, N, D = output['trajectory/positions'].shape
    print(f'Number of timeblocks:          {nblocks = }')
    print(f'Configurations per timeblock:  {nconfs = }')
    print(f'Number of particles:           {N = }')
    print(f'Number of spatial dimensions:  {D = }')

      ########################
      ##   Thermodynamics   ##
      ########################

    # Extract thermodynamic data
    U, W, K = gp.ScalarSaver.extract(output, ['U', 'W', 'K'], per_particle=False, first_block=1)

    # Get the associated times
    times = gp.ScalarSaver.get_times(output, first_block=1)

    # Plot potential energy per particle as a function of time
    plt.figure()
    plt.plot(times*cf['in_ps'], U/N*cf['in_eV'])
    u_lammps = 3.580  # eV/particle
    plt.axhline(u_lammps, color='k', linestyle='--')
    plt.xlabel(r'Time, $t$ [ps]')
    plt.ylabel('Potential energy per particle, $u=U/N$ [eV]')
    plt.savefig('energy.png')
    plt.show()

    # Plot pressure in Gpa
    simbox = output['initial_configuration'].attrs['simbox_data']
    volume = np.prod(simbox)
    rho = N/volume
    dof = D * N - D  # degrees of freedom
    T_kin = 2 * K / dof
    P = rho * T_kin + W / volume

    plt.figure()
    plt.plot(times*cf['in_ps'], P*cf['in_GPa'])
    p_lammps = 71.03  # GPa
    plt.axhline(p_lammps, color='k', linestyle='--')
    plt.xlabel(r'Time, $t$ [ps]')
    plt.ylabel('Pressure, $P$ [GPa]')
    plt.savefig('pressure.png')
    plt.show()

    # Plot WU scatter plot
    plt.figure()
    x = U*cf['in_eV']/N
    y = W*cf['in_eV']/N
    R = np.corrcoef(x, y)[0,1]
    slope = np.cov(x, y)[0,1]/(np.var(x))
    intersection = np.mean(y)-slope*np.mean(x)
    x_fit = np.linspace(np.min(x), np.max(x), 2)
    plt.title(f'R: {R:.4f},   slope = {slope:.3f}')
    plt.plot(x, y, 'o')
    plt.plot(x_fit, slope*x_fit+intersection, 'r--', lw=4.0, alpha=0.8)
    plt.xlabel(r'Potential energy, $U/N$ [eV]')
    plt.ylabel('Virial, $W/N$ [eV]')
    plt.savefig('UW.png')
    plt.show()

      #############
      ##   RDF   ##
      #############

    configuration = gp.Configuration(D=D, N=N)
    configuration.simbox = gp.Orthorhombic(D, output['initial_configuration'].attrs['simbox_data'])
    configuration.ptype = output['initial_configuration/ptype']
    configuration.copy_to_device()

    calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=1000)
    positions = output['trajectory/positions']  # Shape: nblocks, nconfs, N, D
    first_block = 1
    for pos in positions[first_block:, -1, :, :]:
        configuration['r'] = pos
        configuration.copy_to_device()
        calc_rdf.update()
    rdf_data = calc_rdf.read()

    plt.figure()
    plt.plot(rdf_data['distances']*cf['in_Angstrom'], rdf_data['rdf'][:,0,0])
    plt.xlabel(r'Pair distance, $r_{ij}$ [Å]')
    plt.ylabel('Radial Distribution Function')
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.savefig('rdf.png')
    plt.show()

      #############
      ##   MSD   ##
      #############

    qvalues = 3.0/cf['in_Angstrom']
    dynamics = gp.tools.calc_dynamics(output, first_block=first_block, qvalues=qvalues)  # Dictionary with dynamics
    dynamics.keys()

    plt.figure()
    plt.plot(dynamics['times']*cf['in_ps'], dynamics['msd']*cf['in_Angstrom']**2, 'o')
    plt.xscale('log')
    plt.ylim(0, None)
    plt.xlabel(r'Time, $t$ [ps]')
    plt.ylabel(r'Mean squared displacement [Å$^2$]')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.001, 1e2)
    plt.ylim(0.0001, 1e3)
    plt.savefig('msd.png')
    plt.show()

if __name__ == "__main__":
    main()
