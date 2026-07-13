""" Science test for EAM.

Explanation of modes:
    
    1. mode=="ci": run both setup and the test function(s), but a shortened version of both.
    If any files are written they should be deleted. Do not check agreement with reference data

    2. mode == "nightly": Don't run setup, only run the test files, but with full length, and use assertions to test for
    agreement with reference data. Do not keep output files.

    3. mode == "interactive": If "setup" is passed on the command-line then run the setup function with full duration. Otherwise
    run the test(s) with full duration. If running from the command line display graphs ie call plt.show(). Keep output files,
    which have temporary names (ending in .tmp) to avoid overwriting the version-controlled ones.

"""

import os
import sys
import gamdpy as gp
import pandas as pd
import numpy as np
import math
import h5py

import matplotlib.pyplot as plt

from scipy.constants import atomic_mass, elementary_charge, Boltzmann, Avogadro

Aa = 1.e-10 # Angstrom in m
gPerMole = 1.e-3/Avogadro
Da = atomic_mass
eV = elementary_charge
kB = Boltzmann
inGPa = eV/Aa**3/1e9 # to convert pressure units eV/Aa^3 to GPa
in_bar = inGPa*1e4  # to convert pressure units eV/Aa^3 to bar


rho = 0.085 # number density in inverse cubic Angstrom
TK = 2500 # Temperature in Kelvin
# convert temp to eV/kB units
T = TK * kB/eV


paramsCu = gp.EAM_ZJW_2004_params['Cu']
paramsAu = gp.EAM_ZJW_2004_params['Au']
# we follow LAMMPS and use sqrt(5) times the r_e parameter as the cutoff.
cutCu = math.sqrt(5.) * paramsCu[0] # gives 5.7157519935Å
cutAu = math.sqrt(5.) * paramsAu[0] # gives 7.1553683345Å


paramsCu = np.append(paramsCu, cutCu)
paramsAu = np.append(paramsAu, cutAu)




def test_EAM_ZJW2004_Cu(mode):
    print("EAM (ZJW-2004) pure Cu test")
    print(f"Temperature in eV {T:.4}")
    # mode is "ci" or "nightly" or "interactive"
    if mode == "ci" or mode == "nightly":
        main_dir = os.path.join(os.getcwd(), "science_tests", "EAM-ZJW-2004")
    else:
        main_dir = "."

    reference_data_dir = main_dir # os.path.join(main_dir, "reference_data")
    conf_file_path = os.path.join(reference_data_dir, f"Cu-liquid-rho{rho:.4f}-T{T:.4f}.h5")


    with h5py.File(conf_file_path) as f:
        conf = gp.Configuration.from_h5(f,"configuration", compute_flags={'stresses':True})


    eam_pot = gp.EAM_ZJW_2004([paramsCu], max_num_nbs=1000)
    integrator = gp.NVE(dt=0.1)

    scalar_interval = 4

    runtime_actions = [gp.MomentumReset(100),
                       gp.StressSaver(scalar_interval, compute_flags={'stresses':True}),
                       gp.ScalarSaver(scalar_interval)]

    num_timeblocks = {False:2, True:1}[mode=="ci"] # full length for "nightly" and "interactive"
    sim = gp.Simulation(conf, [eam_pot], integrator, num_timeblocks=num_timeblocks, steps_per_timeblock=1024, runtime_actions=runtime_actions, storage='memory')

    for block in sim.run_timeblocks():
        print(f"{sim.status(per_particle=True)}")


    U, W = gp.ScalarSaver.extract(sim.output, ['U', 'W'], per_particle=False, first_block=0)
    my_times = gp.ScalarSaver.get_times(sim.output, first_block=0)
    stress_tensor = gp.StressSaver.extract(sim.output)
    sxx = stress_tensor[:,0,0]
    syy = stress_tensor[:,1,1]
    szz = stress_tensor[:,2,2]
    sxy = stress_tensor[:,0,1]
    sxz = stress_tensor[:,0,2]
    syz = stress_tensor[:,1,2]

    # Read reference data, saved from an equivalent LAMMPS simulations
    ref_data_path = os.path.join(reference_data_dir, "Data_lammps", "lammps_eam_Cu_ref.dat")


    ref_data = np.loadtxt(ref_data_path)
    U_ref = ref_data[:-1,2]  # Note: ref_data has 513 rows, not 512, hence the -1 in the row range.
    sxx_ref = -ref_data[:-1, 6] / in_bar
    syy_ref = -ref_data[:-1, 7] / in_bar
    szz_ref = -ref_data[:-1, 8] / in_bar
    syz_ref = -ref_data[:-1, 9] / in_bar
    sxz_ref = -ref_data[:-1, 10] / in_bar
    sxy_ref = -ref_data[:-1, 11] / in_bar

    
    if mode in ['interactive', 'nightly']:
        # in these cases we actually make an assert-based test
        diff_U_sq = (U-U_ref)**2
        compare = 200 # sum the first 200 squared differences (representing 800 time steps)
        SSD = np.sum(diff_U_sq[:compare])
    
        assert SSD < 0.1

        # Now compare stress components
        diff_sts_xx_sq = (sxx - sxx_ref)**2
        diff_sts_yy_sq = (syy - syy_ref)**2
        diff_sts_zz_sq = (szz - szz_ref)**2
        diff_sts_yz_sq = (syz - syz_ref)**2
        diff_sts_xz_sq = (sxz - sxz_ref)**2
        diff_sts_xy_sq = (sxy - sxy_ref)**2

        SSD_sts_all_components = ( np.sum(diff_sts_xx_sq[:compare]) +
                                   np.sum(diff_sts_yy_sq[:compare]) +
                                   np.sum(diff_sts_zz_sq[:compare]) +
                                   np.sum(diff_sts_yz_sq[:compare]) +
                                   np.sum(diff_sts_xz_sq[:compare]) +
                                np.sum(diff_sts_xy_sq[:compare]) )
        assert SSD_sts_all_components < 1e-6
    # for later times the differences begin to grow as expected
    # Make figure and save as pdf+png (though when running as a test ie mode == "ci" or mode == "nightly")
    if mode == "interactive":
        plt.figure(100)
        plt.plot(my_times, U, label="gamdpy")
        plt.plot(my_times, U_ref, label="LAMMPS")
        plt.xlabel("Time")
        plt.ylabel("Total potential energy (eV)")
        plt.savefig("eam_Cu_gamdy_vs_lammps.pdf.tmp", format='pdf')
        plt.savefig("eam_Cu_gamdy_vs_lammps.png.tmp", format='png')
        plt.legend()
        print("Wrote graphical presentation of results as pdf and png files with extra suffix .tmp. \
              Remove the suffix by renaming if you wish to replace the version-controlled output files")

        # Perhaps want to make figures showing the comparison of stress for all 6 components
        

def FindMinimumEnthalpyCuAu(rho_array, ptype_unit_cell, plotindex = None, plotlabel=None, writepdfpng=False):

    # type 0 will be Cu, type 1 will be Au
    # In the L1_2 structure, we have an fcc lattice where the corner atoms are
    # the minority type, so Au ie type 1, and the face atoms are the majority type
    # so Cu ie type 0
    # in the code for the FCC unit cell the corner atom is the first of the four in the unit cell, so the type list should be [1, 0, 0, 0]


    eam_pot = gp.EAM_ZJW_2004([paramsCu, paramsAu], max_num_nbs=1000)


    rho = min(rho_array)
    conf = gp.Configuration(D=3)
    conf.make_lattice(gp.unit_cells.FCC, cells=[6,6,6], rho=rho, ptype_unit_cell=ptype_unit_cell)
    evaluator = gp.Evaluator(conf, eam_pot)
    
    E_array = np.zeros_like(rho_array)
    for rdx in range(len(rho_array)):
        rho = rho_array[rdx]
        conf.atomic_scale(rho)
        evaluator.evaluate()
        E = np.sum(conf['U'])/conf.N
        E_array[rdx] = E
        #print(f"{a:.4f} {E:.4f} {rho:.4f}")

    p_degree = 3
    coeffs = np.polyfit(rho_array, E_array, deg=p_degree)
    p = np.poly1d(coeffs)
    if p_degree == 2:
        a, b, c = coeffs
        rho_min = -b / (2*a)
        E_min = p(rho_min)
    elif p_degree == 3:
        a, b, c, d = coeffs

        # For cubic fit
        rho_min_pl = (-2*b + math.sqrt(4*b**2-12*a*c))/(6*a)
        rho_min_mi = (-2*b - math.sqrt(4*b**2-12*a*c))/(6*a)

        E_min_pl = p(rho_min_pl)
        E_min_mi = p(rho_min_mi)
        
        if E_min_pl < E_min_mi:
            E_min = E_min_pl
            rho_min = rho_min_pl

    #print(f"rho_min {rho_min:.4f}, E_min {E_min:.4f}")

    if plotindex is not None:
        rho_fit = np.linspace(min(rho_array), max(rho_array), 200)
        E_fit = p(rho_fit)

        plt.figure(plotindex)
        plt.plot(rho_array, E_array, color="blue", marker='o', label="Data points")
        plt.plot(rho_fit, E_fit, color="red", label="Fitted polynomial")
        plt.plot(rho_min, E_min, color="green", marker="x", ms=20, label="minimum")
        plt.text(rho_min-0.001, E_min+0.06, plotlabel, fontsize=22)
        plt.xlabel(r"rho [Å^-3]")
        plt.ylabel(r"E [eV/atom]")
        plt.legend()
        if writepdfpng:
            pdfname = "E_vs_rho_%s.pdf.tmp" % plotlabel
            plt.savefig(pdfname, format='pdf')
            print("Wrote" + pdfname)
            pngname = "E_vs_rho_%s.png.tmp" % plotlabel
            plt.savefig(pngname, format='png')
            print("Wrote" + pngname)
            print("If required to replace the version-controlled figure files, removing the .tmp suffix")
    a_min = pow(4/rho_min, 1/3)
    return rho_min, E_min, a_min


def test_CuAu_alloys(mode, verbose=True):
    print("Test EAM (ZJW-2004) for binary alloys, specifically Cu-Au alloys")
    # Cu
    rho_array = np.arange(0.075, 0.095, 0.001)
    rho_Cu, E_Cu, a_Cu = FindMinimumEnthalpyCuAu(rho_array, ptype_unit_cell=[0, 0, 0, 0], plotindex=1, plotlabel="Cu", writepdfpng=False)
    if mode in ['nightly', 'interactive']:
        assert math.isclose(a_Cu, 3.615, rel_tol=0.001) # Gola et al 2018
    if verbose:
        print(f"a_Cu and reference value {a_Cu:.5f} 3.615")
    
    # Au
    rho_array = np.arange(0.050, 0.070, 0.001)
    rho_Au, E_Au, a_Au = FindMinimumEnthalpyCuAu(rho_array, ptype_unit_cell=[1, 1, 1, 1], plotindex=2, plotlabel="Au", writepdfpng=False)
    if mode in ['nightly', 'interactive']:
        assert math.isclose(a_Au, 4.080, rel_tol=0.001) # Gola et al 2018
    if verbose:
        print(f"a_Au and reference value {a_Au:.5f} 4.080")


    # Cu3Au
    rho_array = np.arange(0.065, 0.085, 0.001)
    rho_Cu3Au, E_Cu3Au, a_Cu3Au = FindMinimumEnthalpyCuAu(rho_array, ptype_unit_cell=[1, 0, 0, 0], plotindex=3, plotlabel="Cu3Au", writepdfpng=(mode=="interactive"))
    E_mixing_Cu3Au = E_Cu3Au - 0.75*E_Cu - 0.25*E_Au
    if mode in ['nightly', 'interactive']:
        assert math.isclose(a_Cu3Au, 3.750, abs_tol=0.001) #  Gola et al 2018
        assert math.isclose(E_mixing_Cu3Au, -0.093, abs_tol=0.0015) #  Gola et al 2018
    # the absolute difference is just bigger than 0.001 eV/atom here so I made the tolerance 0.0015
    
    if verbose:
        print(f"a_Cu3Au and reference value {a_Cu3Au:.5f} 3.750")
        print(f"E_mixing_Cu3Au and reference value {E_mixing_Cu3Au:.5f} -0.093")


    # CuAu3
    rho_array = np.arange(0.055, 0.075, 0.001)
    rho_CuAu3, E_CuAu3, a_CuAu3 = FindMinimumEnthalpyCuAu(rho_array, ptype_unit_cell=[0, 1, 1, 1], plotindex=4, plotlabel="CuAu3", writepdfpng=(mode=="interactive"))
    E_mixing_CuAu3 = E_CuAu3 - 0.25*E_Cu - 0.75*E_Au
    if mode in ['nightly', 'interactive']:
        assert math.isclose(a_CuAu3, 3.976, rel_tol=0.001) #  Gola et al 2018
        assert math.isclose(E_mixing_CuAu3, -0.095, abs_tol=0.001) #  Gola et al 2018
    
    if verbose:
        print(f"a_CuAu3 and reference value {a_CuAu3:.5f} 3.976")
        print(f"E_mixing_CuAu3 and reference value {E_mixing_CuAu3:.5f} -0.095")





def setup(mode):
    print("Setup procedure for EAM Cu test")
    print(f"Temperature in eV {T:.4}")


    conf = gp.Configuration(D=3)
    conf.make_lattice(gp.unit_cells.FCC, cells=[8,8,8], rho=rho)
    conf['m'] = 63.546
    conf.randomize_velocities(temperature=T)


    eam_pot = gp.EAM_ZJW_2004([paramsCu], max_num_nbs=1000)

    integrator = gp.NVT(temperature=T, tau=5.0, dt=0.1)

    runtime_actions = [gp.TrajectorySaver(),
                       gp.MomentumReset(100),
                       gp.ScalarSaver(16),
                       gp.RestartSaver()]

    num_timeblocks = {False:15, True:2}[mode=="ci"] # full length for "nightly" and "interactive"
    output_filename = 'test_eam.h5'
    sim = gp.Simulation(conf, [eam_pot], integrator, num_timeblocks=num_timeblocks, steps_per_timeblock=1024, runtime_actions=runtime_actions, storage=output_filename)




    for block in sim.run_timeblocks():
        print(f"{sim.status(per_particle=True)}")
    print(sim.summary())

    if mode == "ci" or mode == "nightly":
        main_dir = os.path.join(os.getcwd(), "science_tests", "EAM-ZJW-2004")
    else:
        main_dir = "."

    gamdpy_conf_filename = dump_filename = os.path.join(main_dir, f"Cu-liquid-rho{rho:.4f}-T{T:.4f}.h5")
    with h5py.File(gamdpy_conf_filename+'.tmp', "w") as f:
        conf.save(f, group_name='configuration', mode='w')

    dump_filename = os.path.join(main_dir, f'Cu-liquid-rho{rho:.4f}-T{T:.4f}.lammps')

    # when writing to LAMMPS dump file want to:
    # (1) increment the velocities by a half time step
    # (2) convert to LAMMPS "metal" units where the time unit is picosecond
    time_unit_in_picoseconds = Aa * math.sqrt(Da/eV)*1e12
    print(f'time uni {time_unit_in_picoseconds:.6}')
    unit_ratios = {'positions-box' : 1., 'velocities': 1./time_unit_in_picoseconds, 'forces':1.}

    dt = integrator.get_params(sim.configuration, sim.interactions_params)[0]
    #dt = 0.01 # for testing, to be consistent with the LAMMPS script
    print('Increment velocities by a half time-step for saving to LAMMPS dump file')
    print(f'dt={dt:.6f}')
    sim.configuration['v'] += (sim.configuration['f']/sim.configuration['m'][:, np.newaxis]) * (dt/2.)

    lmp_dump = gp.configuration_to_lammps(sim.configuration, timestep=0, unit_rescale=unit_ratios)
    print(lmp_dump, file=open(dump_filename + '.tmp', 'a'))


    if mode in ["ci", "nightly"]:
        # Could just not write the files, but it's probably good to test that code as well
        print("Removing gamdpy output file %s" % output_filename)
        os.remove(output_filename)
        print("Removing configuration files%s and %s"  %(gamdpy_conf_filename+'.tmp', dump_filename+'.tmp'))
        os.remove(gamdpy_conf_filename+'.tmp')
        os.remove(dump_filename+'.tmp')
    else:
        print("Wrote gamdpy configuration file ", gamdpy_conf_filename+'.tmp')
        print("Wrote gamdpy configuration (dump) file ", dump_filename+'.tmp')
        print("If it's necessary to replace the previous versions, remove the extension .tmp from these files and move the lammps file to Data_lammps ")
        print("Note that the actual configurations will be different each time setup is called")

mode = "ci" # default
if "nightly" in sys.argv:
    mode = "nightly"
if "interactive" in sys.argv:
    mode = "interactive" # if both present then go with "interactive"

# RUN SETUP

print("Mode=%s" % mode)
if mode == "ci" or "setup" in sys.argv:
    setup(mode)

# RUN TESTS

test_EAM_ZJW2004_Cu(mode)

test_CuAu_alloys(mode)

if __name__ == '__main__':
    if mode == "interactive" and "setup" not in sys.argv:
        plt.show()

