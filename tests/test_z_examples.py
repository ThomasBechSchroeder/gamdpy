import pytest

# These are flags
timing = False  # Timing of examples
rm_out = True   # Removing output created by examples
if timing: fout = open("timing_examples.txt", "w")

@pytest.mark.slow
class Test_examples:
    ''' This test is testing examples '''
    # LC: examples excluded are LJchain_wall.py, minimal_cpu.py

    import sys, os
    import numpy as np
    main_dir = os.getcwd()

    import matplotlib
    matplotlib.use('Agg')  # Static backend that does not halt on plt.show()
    os.environ['MPLBACKEND'] = 'Agg' # Reduced warnings (LC: not anymore, no idea why) 

    # This function get name and make a test for example with that name
    # The function also includes timing
    # The list toremove contains files created from the script which need to be removed
    def make_one(self, name="", toremove=[]):
        import os
        import importlib
        os.chdir(os.path.join(self.main_dir, "examples"))
        if timing:                      # save start time
            import time
            start = time.time()
        importlib.__import__(f"examples.{name}")
        if timing:                      # save end time 
            end = time.time()
            fout.write(f"{name:30s}\t{end-start:>8.2f}\n")
        if rm_out:                      # remove created files
            [os.remove(filename) for filename in toremove]

    #def test_analyze_structure(self): # Test does not work because reading from command-line
    #    self.make_one("analyze_structure", ["Data/LJ_r0.973_T0.70_toread_rdf.pdf", "Data/LJ_r0.973_T0.70_toread_rdf.pkl"])

    # Tests are ordered try to go fast to slow
    def test_D2(self):
        self.make_one("D2")

    def test_bcc_lattice(self):
        self.make_one("bcc_lattice")

    #def test_blocks(self): Example removed
    #    self.make_one("blocks")

    def test_minimal(self):
        self.make_one("minimal")

    def test_minimal_NPT(self):
        self.make_one("minimal_NPT")

    def test_yukawa(self):
        self.make_one("yukawa")

    def test_calc_rdf_from_h5(self):
        self.make_one("calc_rdf_from_h5", ["rdf.dat", "ptype_rdf.dat"])

    def test_calc_rdf_from_rumd3(self):
        self.make_one("calc_rdf_from_rumd3", ["rdf_rumd3.dat", "ptype_rdf_rumd3.dat"])

    def test_calc_sq_from_h5(self):
        self.make_one("calc_sq_from_h5", ["sq.dat"])

    def test_analyze_thermodynamics(self):
        self.make_one("analyze_thermodynamics", ["Data/LJ_r0.973_T0.70_toread_thermodynamics.pdf"])

    def test_analyze_dynamics(self):
        self.make_one("analyze_dynamics", ["Data/LJ_r0.973_T0.70_toread_dynamics.pdf","Data/LJ_r0.973_T0.70_toread_dynamics.pkl"])

    def test_analyze_structure(self):
        self.make_one("analyze_structure", ["Data/LJ_r0.973_T0.70_toread_rdf.pdf","Data/LJ_r0.973_T0.70_toread_rdf.pkl"])

    def test_ASD(self):
        self.make_one("ASD", ["Data/ASD_rho1.863_T0.465.h5"])

    def test_NVU_RT_kob_andersen(self):
        self.make_one("NVU_RT_kob_andersen")

    def test_evaluator_einstein_crystal(self):
        self.make_one("evaluator_einstein_crystal")

    def test_evaluator_inverse_powerlaw(self):
        self.make_one("evaluator_inverse_powerlaw")

    def test_structure_factor(self):
        self.make_one("structure_factor")

    def test_switching_integrator(self):
        self.make_one("switching_integrator")

    def test_tethered_particles(self):
        self.make_one("tethered_particles", ["initial.xyz", "final.xyz"])

    def test_widoms_particle_insertion(self):
        self.make_one("widoms_particle_insertion")

    def test_LJchain(self):
        self.make_one("LJchain", ["Data/LJchain10_Rho1.00_T0.700.h5"])

    def test_write_to_lammps(self):
        self.make_one("write_to_lammps", ["dump.initial", "dump.lammps"])

    def test_thermodynamics(self):
        self.make_one("thermodynamics")

    def test_generic_molecules(self):
        self.make_one("generic_molecules")

    def test_LJ(self):
        self.make_one("LJ")
    
    def test_D4(self):
        self.make_one("D4")

    def test_D8(self):
        self.make_one("D8")

    def test_poiseuille(self):
        self.make_one("poiseuille", ["HydrodynamicProfile.dat", "initial.xyz", "final.xyz"])

    def test_isochore(self):
        self.make_one("isochore", ["Data/LJ_r0.973_T0.70.h5", "Data/LJ_r0.973_T1.10.h5", "Data/LJ_r0.973_T1.50.h5"])

    def test_shear_SLLOD(self):
        self.make_one("shear_SLLOD", ["shear_run.txt"])

    def test_hydrocorr(self):
        self.make_one("hydrocorr", ["jacf.dat", "dacf.dat"])

    def test_isomorph(self):        # Note: this script produces Data/isomorph.pkl needed by next two. File gets remove in next
        self.make_one("isomorph")

    def test_plot_isomorph_rdf(self):        
        self.make_one("plot_isomorph_rdf", ["isomorph_rdf.pdf"])

    def test_plot_isomorph_dynamics(self):
        self.make_one("plot_isomorph_dynamics", ["Data/isomorph.pkl", "isomorph_dynamics.pdf"])

    def test_consistency_NPT(self):
        self.make_one("consistency_NPT")

    def test_kablj(self):
        self.make_one("kablj", ["Data/KABLJ_Rho1.200_T0.800.h5"])

    def test_molecules(self):
        self.make_one("molecules", ["Data/molecules.h5", "Data/dump.lammps", "Data/dump_compress.lammps", "molecule.pdf"])

    def test_rubber_cube(self):
        self.make_one("rubber_cube", ['Data/rubber_cube.lammps'])


 
