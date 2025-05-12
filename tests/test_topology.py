def test_topology():
    import gamdpy as gp
    import numpy as np
    import random

    bond_length = 0.9
    cut_off=bond_length*1.1
    num_particles = 270
    positions = []

    factor = (1/3)**(1/2)*bond_length

    for i in range(num_particles):
        positions.append( [ i*factor, i*factor, i*factor ] ) # x, y, z for this particle
    
    random.shuffle(positions)

    bonds = gp.bonds_from_positions(positions=positions, cut_off=cut_off, bond_type=0)
    assert len(bonds) == num_particles-1, f'Did not find the correct number of bonds:  {len(bonds)=} != {num_particles-1=}'
    # add test: duplicated bonds, distance
    
    bonds = [ [bond[1], bond[0], bond[2]]  if random.choice([False, True]) else bond for bond in bonds] # revert ~half bonds
    bonds_without_type = [ bond[0:2] for bond in bonds ]

    angles = gp.angles_from_bonds(bonds=bonds, angle_type=0)
    assert len(angles) == num_particles-2, f'Incorrect number of angles:  {len(angles)=} != {num_particles-2=}'
    for index, angle in enumerate(angles):
        assert angle not in angles[index+1:], f'Duplicated angle found: {angle=}, \n{angles[index+1:]=}'
        reverse = [angle[2], angle[1], angle[0], angle[3]]
        assert reverse not in angles[index+1:], f'Duplicated angle (reversed) found: {angle=}, \n{angles[index+1:]=}'
    for angle in angles:
        bond0 = angle[0:2]
        reverse0 = bond0[::-1]
        bond1 = angle[1:3]
        reverse1 = bond1[::-1]
        assert bond0 != bond1 and bond0 != reverse1, f'Angle with duplicate bonds found {angle=}'
        assert bond0 in bonds_without_type or reverse0 in bonds_without_type, f'Angle has bond not found in bonds \n{angle=}, \n{bonds=}, \n{bonds_without_type=}, \n{bond0=}'
        assert bond1 in bonds_without_type or reverse1 in bonds_without_type, f'Angle has bond not found in bonds \n{angle=}, \n{bonds=}, \n{bonds_without_type=}, \n{bond1=}'
    


    angles = [ [angle[2], angle[1], angle[0], angle[3]]  if random.choice([False, True]) else angle for angle in angles] # revert ~half dihedrals
    angles_without_type = [ angle[0:3] for angle in angles ]

    dihedrals = gp.dihedrals_from_angles(angles=angles, dihedral_type=0)
    assert len(dihedrals) == num_particles-3, f'Incorrect number of dihedrals:  {len(dihedrals)=} != {num_particles-3=}'
    for index, dihedral in enumerate(dihedrals):
        assert dihedral not in dihedrals[index+1:], f'Duplicated dihedrals found: {dihedral=}, \n{dihedrals[index+1:]=}'
        reverse = [dihedral[3], dihedral[2], dihedral[1], dihedral[0], dihedral[4]]
        assert reverse not in dihedrals[index+1:], f'Duplicated dihedral (reversed) found: {dihedral}, \n{dihedrals[index+1:]=}'
    for dihedral in dihedrals:
        angle0 = dihedral[0:3]
        reverse0 = angle0[::-1]
        angle1 = dihedral[1:4]
        reverse1 = angle1[::-1]
        assert angle0 != angle1 and angle0 != reverse1, f'Dihedral with duplicate angles found {dihedral=}'
        assert angle0 in angles_without_type or reverse0 in angles_without_type, f'Dihedral has angle not found in angles \n{dihedral=}, \n{angles=}, \n{angles_without_type=}, \n{angle0=}'
        assert angle1 in angles_without_type or reverse1 in angles_without_type, f'Dihedral has angle not found in angles \n{dihedral=}, \n{angles=}, \n{angles_without_type=}, \n{angle1=}'
    

if __name__ == '__main__':
    test_topology()
