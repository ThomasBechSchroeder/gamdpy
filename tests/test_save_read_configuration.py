def test_save_read_configuration():
    import gamdpy as gp
    import numpy as np
    import numpy.testing as npt
    import random
    import h5py

    # Simulation params 
    rho, temperature = 0.85, 1.5
    N_A, N_B, N_C = 8, 4, 4  # Number of atoms of each tyoe
    particles_per_molecule = N_A + N_B + N_C
    num_timeblocks = 64
    steps_per_timeblock = 1 * 1024 # 8 * 1024 to show reliable pattern formation

    positions = []
    particle_types = []
    masses = []

    # A particles
    for i in range(N_A):
        positions.append( [ i*1.0, (i%2)*.1, 0. ] ) # x, y, z for this particle
        particle_types.append( 0 )
        masses.append( 1.0 )  

    # B particles
    for i in range(N_B):
        positions.append( [ 0, (i+1)*1.0, ((i+1)%2)*.1 ] ) # x, y, z for this particle
        particle_types.append( 1 )
        masses.append( 1.0 )  

    # C particles
    for i in range(N_C):
        positions.append( [ ((i+1)%2)*.1, 0, (i+1)*1.0 ] ) # x, y, z for this particle
        particle_types.append( 2 )
        masses.append( 1.0 )  

    # Setup configuration: Single molecule first, then duplicate
    top = gp.Topology(['MyMolecule', ])
    top.bonds = gp.bonds_from_positions(positions, cut_off=1.1, bond_type=0)
    top.angles = gp.angles_from_bonds(top.bonds, angle_type=0)
    top.dihedrals = gp.dihedrals_from_angles(top.angles, dihedral_type=0)
    top.molecules['MyMolecule'] = gp.molecules_from_bonds(top.bonds)


    dict_this_mol = {"positions" : positions,
                     "particle_types" : particle_types,
                     "masses" : masses,
                     "topology" : top}



    configuration = gp.replicate_molecules([dict_this_mol], [216], safety_distance=2.0, compute_flags={"stresses":True})

    configuration.randomize_velocities(temperature=temperature)


    filename = 'test_conf.h5'
    group_name = 'test_configuration'

    with h5py.File(filename,'w') as f:
        configuration.save(f, group_name)

    # Now open the file for reading and read into another configuration
    with h5py.File(filename,'r') as f:
        conf_from_file = gp.Configuration.from_h5(f, group_name, include_topology=True)


    npt.assert_array_equal(conf_from_file['r'], configuration['r'])
    npt.assert_array_equal(conf_from_file['v'], configuration['v'])
    npt.assert_array_equal(conf_from_file['m'], configuration['m'])
    npt.assert_array_equal(conf_from_file.ptype, configuration.ptype)
    npt.assert_array_equal(conf_from_file.r_im, configuration.r_im)


    npt.assert_array_equal(configuration.scalars, conf_from_file.scalars)

    # sim box
    assert conf_from_file.simbox.get_name() == configuration.simbox.get_name()
    npt.assert_array_equal(conf_from_file.simbox.data_array, configuration.simbox.data_array)
    
    # topology
    npt.assert_array_equal(np.array(conf_from_file.topology.bonds),
                           np.array(configuration.topology.bonds))

    npt.assert_array_equal(np.array(conf_from_file.topology.angles),
                           np.array(configuration.topology.angles))

    npt.assert_array_equal(np.array(conf_from_file.topology.dihedrals),
                           np.array(configuration.topology.dihedrals))

    for key in configuration.topology.molecules:
        npt.assert_array_equal(conf_from_file.topology.molecules[key], configuration.topology.molecules[key])


if __name__ == '__main__':
    test_save_read_configuration()
