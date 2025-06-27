
def test_exclusions():
    import numpy as np
    import gamdpy as gp

    nmols = 2
    r0=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    mass=[1.0, 1.0, 1.0, 1.0]
    types=[0, 0, 0, 0]

    top = gp.Topology(['test_mol', ])
    top.bonds = gp.bonds_from_positions(r0, cut_off=1.1, bond_type=0)
    top.angles = gp.angles_from_bonds(top.bonds, angle_type=0)
    top.dihedrals = gp.dihedrals_from_angles(top.angles, dihedral_type=0)
    top.molecules['test_mol'] = gp.molecules_from_bonds(top.bonds)

    dict_test_mol = {"positions" : r0, "particle_types" : types, "masses" : mass, "topology" : top}

    configuration = gp.replicate_molecules([dict_test_mol], [nmols], safety_distance=3.0)

    bonds = gp.Bonds(gp.harmonic_bond_function, [1.0, 1.0], configuration.topology.bonds)
    angles = gp.Angles(gp.cos_angle_function, configuration.topology.angles, [1.0, 1.0])
    dihedrals = gp.Dihedrals(gp.ryckbell_dihedral, configuration.topology.dihedrals, 
                             parameters=[.0, 5.0, .0, .0, .0, .0])

    for option in ['bonds', 'angles', 'dihedrals']:
        if option=='bonds':
            exclusions = bonds.get_exclusions(configuration)
            answer = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 2]])
        elif option=='angles':
            exclusions = angles.get_exclusions(configuration)
            answer = np.array([[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                                [0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]])        
        elif option=='dihedrals': 
            exclusions = dihedrals.get_exclusions(configuration)
            answer = np.array([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                               [0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]])

        assert np.array(exclusions[:1][:]).all() == answer.all()

       
if __name__ == "__main__":
    test_exclusions()

