import pytest
from gamdpy.configuration.topology import molecules_from_bonds

def test_simple_separation():
    """Test that two separate molecules are identified and sorted by first atom."""
    bonds = [[2, 3], [0, 1]]
    result = molecules_from_bonds(bonds)
    # The outer list is sorted by the first atom (0 < 2)
    assert result == [[0, 1], [2, 3]]

def test_bond_metadata_handling():
    """Verify metadata is ignored and outer list is sorted by the first atom."""
    bonds = [
        [4, 5, "aromatic"],
        [0, 1, "single", 1.54], 
        [1, 2, "double", 1.34]
    ]
    result = molecules_from_bonds(bonds)
    # Expected order: [0, 1, 2] comes before [4, 5]
    assert result == [[0, 1, 2], [4, 5]]

def test_internal_and_outer_sorting():
    """Verify both internal atom sorting and outer molecule sorting."""
    # Input has jumbled atom pairs and jumbled molecule order
    bonds = [[20, 15], [5, 2]]
    result = molecules_from_bonds(bonds)
    
    # Molecule 1: [2, 5], Molecule 2: [15, 20]
    # Outer sort: [2, 5] index 0 < [15, 20] index 0
    assert result == [[2, 5], [15, 20]]

def test_complex_bridge_merge():
    """Ensure the 'Bridge Problem' resolves and returns a clean, sorted list."""
    bonds = [
        [5, 6], [7, 8], # Group B (later)
        [1, 2], [3, 4], # Group A (earlier)
        [2, 3],         # Bridge for A
        [6, 7]          # Bridge for B
    ]
    result = molecules_from_bonds(bonds)
    
    expected = [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]
    assert result == expected

def test_overlapping_initial_indices():
    """Test sorting when first atoms are close."""
    bonds = [[1, 10], [0, 5]]
    result = molecules_from_bonds(bonds)
    assert result == [[0, 5], [1, 10]]

def test_empty_input():
    """Ensure the function handles an empty bond list gracefully."""
    assert molecules_from_bonds([]) == []

@pytest.mark.parametrize("input_bonds, expected", [
    ([[0, 1], [1, 2]], [[0, 1, 2]]),
    ([[2, 3], [0, 1], [4, 5]], [[0, 1], [2, 3], [4, 5]]),
    ([[0, 5], [0, 1]], [[0, 1, 5]]),
])
def test_molecule_exact_match(input_bonds, expected):
    """Verify exact list equality for various scenarios."""
    assert molecules_from_bonds(input_bonds) == expected
