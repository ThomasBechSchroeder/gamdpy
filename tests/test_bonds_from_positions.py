import pytest
import numpy as np
from gamdpy import bonds_from_positions  

def test_single_bond():
    """Test two atoms within the cutoff distance."""
    positions = [[0, 0, 0], [1, 0, 0]]
    cut_off = 1.1
    bond_type = "single"
    result = bonds_from_positions(positions, cut_off, bond_type)
    
    assert result == [[0, 1, "single"]]

def test_no_bond_outside_cutoff():
    """Test two atoms just outside the cutoff distance."""
    positions = [[0, 0, 0], [2, 0, 0]]
    cut_off = 1.0
    result = bonds_from_positions(positions, cut_off, "single")
    
    assert result == []

def test_exact_cutoff_distance():
    """Test atoms exactly at the cutoff distance (should be included)."""
    # Distance is exactly 1.0
    positions = [[0, 0, 0], [1, 0, 0]]
    cut_off = 1.0
    result = bonds_from_positions(positions, cut_off, "boundary")
    
    assert len(result) == 1
    assert result[0] == [0, 1, "boundary"]

def test_multiple_bonds():
    """Test a 3-atom chain where 0-1 and 1-2 are bonded, but 0-2 is not."""
    positions = [
        [0, 0, 0],  # Index 0
        [1, 0, 0],  # Index 1
        [2, 0, 0]   # Index 2
    ]
    cut_off = 1.5
    result = bonds_from_positions(positions, cut_off, 1)
    
    # Expect bonds: [0, 1, 1] and [1, 2, 1]
    # Note: the function appends [j, i] where j < i
    expected = [[0, 1, 1], [1, 2, 1]]
    assert result == expected

def test_empty_or_single_atom():
    """Test that no bonds are formed with insufficient atoms."""
    assert bonds_from_positions([], 1.0, "type") == []
    assert bonds_from_positions([[0, 0, 0]], 1.0, "type") == []

def test_3d_distance_calculation():
    """Test distance calculation across all three axes."""
    # Distance is sqrt(1^2 + 1^2 + 1^2) = sqrt(3) approx 1.732
    positions = [[0, 0, 0], [1, 1, 1]]
    
    # Should not bond at cutoff 1.7
    assert bonds_from_positions(positions, 1.7, "3d") == []
    # Should bond at cutoff 1.8
    assert len(bonds_from_positions(positions, 1.8, "3d")) == 1

# --- 2D Space Tests ---

def test_2d_diagonal_bond():
    """Test distance logic in 2D (Pythagorean triple 3-4-5)."""
    # Distance = sqrt(3^2 + 4^2) = 5.0
    positions = [
        [0, 0],
        [3, 4]
    ]
    
    # Just at the limit
    assert len(bonds_from_positions(positions, 5.0, "2D")) == 1
    # Just below the limit
    assert len(bonds_from_positions(positions, 4.9, "2D")) == 0

def test_2d_multiple_neighbors():
    """Test multiple connections in a 2D grid."""
    positions = [
        [0, 0], # 0
        [1, 0], # 1
        [0, 1]  # 2
    ]
    # Cutoff of 1.1 should catch horizontal and vertical, but not diagonal (sqrt(2))
    result = bonds_from_positions(positions, 1.1, "flat")
    
    expected = [[0, 1, "flat"], [0, 2, "flat"]]
    assert sorted(result) == sorted(expected)

# --- 4D Space Tests ---

def test_4d_hyperspace_bond():
    """Test distance logic in 4D (x, y, z, w)."""
    # Distance = sqrt(1^2 + 1^2 + 1^2 + 1^2) = sqrt(4) = 2.0
    positions = [
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ]
    
    # Should bond at cutoff 2.0
    assert len(bonds_from_positions(positions, 2.0, "4D")) == 1
    # Should not bond at cutoff 1.9
    assert len(bonds_from_positions(positions, 1.9, "4D")) == 0

def test_4d_unit_interval():
    """Verify that only specific dimensions triggering the cutoff works."""
    positions = [
        [0, 0, 0, 0],
        [0, 0, 0, 5]  # Only the 'w' dimension has distance
    ]
    
    assert len(bonds_from_positions(positions, 5.0, "hyper")) == 1
    assert len(bonds_from_positions(positions, 4.9, "hyper")) == 0