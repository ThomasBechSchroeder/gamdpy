import pytest
from gamdpy import dihedrals_from_angles

@pytest.fixture
def dihedral_type():
    return "OPLS_123"

@pytest.mark.parametrize("angles, expected_indices", [
    # Case 1: Standard head-to-tail overlap
    ([[0, 1, 2], [1, 2, 3]], [0, 1, 2, 3]),
    
    # Case 2: Tail-to-tail overlap (second angle reversed)
    ([[0, 1, 2], [3, 2, 1]], [0, 1, 2, 3]),
    
    # Case 3: Head-to-head overlap (first angle reversed)
    ([[2, 1, 0], [1, 0, 3]], [2, 1, 0, 3]),
    
    # Case 4: Head-to-tail overlap (first angle reversed, second angle reversed)
    ([[2, 1, 0], [3, 0, 1]], [2, 1, 0, 3]),
])
def test_overlap_permutations(angles, expected_indices, dihedral_type):
    """Tests all four logical branches for atom connectivity."""
    result = dihedrals_from_angles(angles, dihedral_type)
    assert len(result) == 1
    assert result[0] == expected_indices + [dihedral_type]

def test_no_matches(dihedral_type):
    """Ensures empty list is returned when no angles share a bond."""
    angles = [[0, 1, 2], [3, 4, 5]]
    assert dihedrals_from_angles(angles, dihedral_type) == []

def test_long_chain(dihedral_type):
    """Tests a 5-atom chain (0-1-2-3-4) which should produce 2 dihedrals."""
    angles = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    result = dihedrals_from_angles(angles, dihedral_type)
    
    expected = [
        [0, 1, 2, 3, dihedral_type],
        [1, 2, 3, 4, dihedral_type]
    ]
    # Check length and content regardless of order
    assert len(result) == 2
    for item in expected:
        assert item in result