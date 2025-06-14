"""Test the functions which saves/load configurations. """

def test_savers():
    import os
    import gamdpy as gp
    import numpy as np
    import h5py
    # Create a configuration to save
    conf = gp.Configuration(D=3)
    conf.make_positions(N=10, rho=1.0)
    # Save as xyz
    gp.tools.save_configuration(configuration=conf, filename="final.xyz", format="xyz")
    a = np.loadtxt("final.xyz", skiprows=1)
    assert (a==np.c_[conf.ptype, conf['r']]).all(), "Failure in save_configuration"
    os.remove("final.xyz")
    # Save as h5
#    gp.configuration_to_hdf5(configuration=conf, filename="final.h5")
#    with h5py.File("final.h5", "r") as f:
#        assert [key for key in f.keys()] == ['m', 'ptype', 'r', 'r_im', 'v']
#        assert (f['r'] == conf['r']).all()
#        assert (f['v'] == conf['v']).all()
#        assert (f['ptype'] == conf.ptype).all()
#        assert (f['m'] == conf['m']).all()
#    os.remove("final.h5")
    # Save as lammps
    assert isinstance(gp.configuration_to_lammps(configuration=conf), str)
    # Save/load to/from rumd3
    gp.configuration_to_rumd3(configuration=conf, filename="restart.xyz.gz")
    read_conf = gp.configuration_from_rumd3("restart.xyz.gz")
    assert (read_conf['r'] == conf['r']).all()
    assert (read_conf['v'] == conf['v']).all()
    os.remove("restart.xyz.gz")

if __name__ == '__main__':
    test_savers()
