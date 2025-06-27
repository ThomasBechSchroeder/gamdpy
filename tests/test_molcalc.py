'''
    Tests molecular calculators
'''

def test_molprop():
    import numpy as np
    import gamdpy as gp

    nmols = 10
    r0=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    mass=[1.0, 0.5, 0.25]
    types=[0, 0, 0]
    charges=[1, 0, -1]

    top = gp.Topology(['test_mol', ])
    top.bonds = gp.bonds_from_positions(r0, cut_off=1.1, bond_type=0)
    top.molecules['test_mol'] = gp.molecules_from_bonds(top.bonds)

    dict_test_mol = {"positions" : r0, "particle_types" : types, "masses" : mass, "topology" : top}

    configuration = gp.replicate_molecules([dict_test_mol], [nmols], safety_distance=3.0)
    configuration.randomize_velocities(temperature=1.0, seed=1216)

    rmol, mmol = gp.tools.calculateMolCenterMass(configuration, 'test_mol')
    vmol = gp.tools.calculateMolVelocity(configuration, 'test_mol')
    dmol = gp.tools.calculateMolDipole(configuration, charges, 'test_mol') [0]

    dmag = np.sqrt(dmol[0,0]**2 + dmol[0,1]**2 + dmol[0,2]**2)

    _range = configuration.simbox.get_lengths()*0.5
    
    for k in range(3):
        assert -_range[k] < np.min(rmol[:, k]) + 1e-6
        assert _range[k] > np.max(rmol[:,k]) - 1e-6

    assert np.abs(np.sum(np.sum(vmol))) < 1e-6
    assert np.sum( mmol ) == nmols*np.sum( mass )
    assert np.abs(dmag - 2.0) < 1e-6

if __name__ == '__main__':
    
    test_molprop()







