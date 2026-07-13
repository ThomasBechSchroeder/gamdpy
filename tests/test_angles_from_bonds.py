import pytest
from gamdpy import angles_from_bonds

def test_linear_chain():
    """Test a simple 3-atom linear chain (2 bonds)."""
    bonds = [[0, 1], [1, 2]]
    angle_type = "harmonic"
    expected = [[0, 1, 2, "harmonic"]]
    assert angles_from_bonds(bonds, angle_type) == expected

def test_branched_system():
    """Test a central atom bonded to three others (forming 3 angles)."""
    bonds = [[0, 1], [0, 2], [0, 3]]
    angle_type = 1
    result = angles_from_bonds(bonds, angle_type)
    
    # Angles should be (1-0-2), (1-0-3), (2-0-3)
    assert len(result) == 3
    assert [1, 0, 2, 1] in result
    assert [1, 0, 3, 1] in result
    assert [2, 0, 3, 1] in result

def test_connectivity_permutations():
    """Verify all four 'if/elif' logic branches for bond indices."""
    angle_type = "test"
    
    # Case 1: bond[0] == other_bond[0]
    assert [1, 0, 2, "test"] in angles_from_bonds([[0, 1], [0, 2]], angle_type)
    # Case 2: bond[0] == other_bond[1]
    assert [1, 0, 2, "test"] in angles_from_bonds([[0, 1], [2, 0]], angle_type)
    # Case 3: bond[1] == other_bond[0]
    assert [0, 1, 2, "test"] in angles_from_bonds([[0, 1], [1, 2]], angle_type)
    # Case 4: bond[1] == other_bond[1]
    assert [0, 1, 2, "test"] in angles_from_bonds([[0, 1], [2, 1]], angle_type)

def test_no_angles():
    """Test scenarios where no angles should be detected."""
    # Disjoint bonds
    assert angles_from_bonds([[0, 1], [2, 3]], "type") == []
    # Single bond
    assert angles_from_bonds([[0, 1]], "type") == []
    # Empty list
    assert angles_from_bonds([], "type") == []

def test_cyclic_system():
    """Test a 3-atom ring (3 bonds, 3 angles)."""
    bonds = [[0, 1], [1, 2], [2, 0]]
    result = angles_from_bonds(bonds, "ring")
    assert len(result) == 3
    # Check for the three vertices: 1, 2, and 0
    centers = [angle[1] for angle in result]
    assert sorted(centers) == [0, 1, 2]