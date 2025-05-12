def test_Simbox():
    import gamdpy as gp
    import numpy as np

    simbox = gp.Orthorhombic(D=3, lengths=np.array([3,4,5]))
    assert isinstance(simbox, gp.Orthorhombic), "Problem with Simbox __init__"
    assert np.all(simbox.lengths == np.array([3,4,5])), "Problem with Simbox.lengths"

    ## Test dist_sq_dr_function
    ri = np.array([2.5,3,4])
    rj = np.array([0.5,0.5,0.5])
    dr = 0*ri
    result = simbox.get_dist_sq_dr_function()(ri, rj, simbox.lengths, dr)
    assert result == 5.5, "Problem with Simbox.dist_sq_dr_function result" 
    assert np.all( dr == np.array([-1, -1.5, -1.5]) ), "Problem with Simbox.dist_sq_dr_function dr" 

    ## Test dist_sq_function
    result = simbox.get_dist_sq_function()(ri, rj, simbox.lengths)
    assert result == 5.5, "Problem with Simbox.dist_sq_function result" 

    ## Test apply_PBC
    # Here the r is out of the box and PBC are applied by increasing img count
    r, img = np.array([4,5,6]), np.array([0,1,2])
    simbox.get_apply_PBC()(r, img, simbox.lengths)
    assert np.all( r   == np.array([1,1,1])), "Problem with Simbox.apply_PBC:r positive img"
    assert np.all( img == np.array([1,2,3])), "Problem with Simbox.apply_PBC:img positive img"
    # Here the r is inside the box and PBC are applied by decreasing img count
    r, img = -np.array([4,5,6]), np.array([0,1,2])
    simbox.get_apply_PBC()(r, img, simbox.lengths)
    assert np.all( r   == np.array([-1,-1,-1])), "Problem with Simbox.apply_PBC:r negative img"
    assert np.all( img == np.array([-1,0,1])), "Problem with Simbox.apply_PBC:img negative img"

    ## Test volume
    assert simbox.get_volume_function()(simbox.lengths)==3*4*5, "Problem with Simbox.volume"

    ## Test dist_moved_sq_function
    r_current = np.array([1,1,1])
    r_last    = r_current + 0.75*simbox.lengths
    result = simbox.get_dist_moved_sq_function()(r_current, r_last, simbox.lengths, simbox.lengths)
    assert result == 3.125, "Problem with simbox.dist_moved_sq_function"

    ## Test dist_moved_exceeds_limit_function
    r_current = np.array([1,1,1])
    r_last    = r_current + 0.50e0
    skin, cut = 2.e0, 1.e0
    result = simbox.get_dist_moved_exceeds_limit_function()(r_current, r_last, simbox.lengths, simbox.lengths, skin, cut)
    assert result == False, "Problem with Simbox.dist_moved_exceeds_limit_function call 1"
    r_last    = r_current + 0.75e0
    result = simbox.get_dist_moved_exceeds_limit_function()(r_current, r_last, simbox.lengths, simbox.lengths, skin, cut)
    assert result == True, "Problem with Simbox.dist_moved_exceeds_limit_function call 2"

def test_LeesEdwards():
    import pytest
    import gamdpy as gp
    import numpy as np

    # Test 1D error 
    # https://stackoverflow.com/questions/23337471/how-do-i-properly-assert-that-an-exception-gets-raised-in-pytest
    with pytest.raises(ValueError) as e_info:
        simbox = gp.LeesEdwards(D=1, lengths=np.array([3,4]), box_shift=1.0)
    assert e_info.type is ValueError

    # Test normal case
    simbox = gp.LeesEdwards(D=3, lengths=np.array([3,4,5]), box_shift=1.0)
    assert isinstance(simbox, gp.LeesEdwards), "Problem with simbox __init__"
    assert np.all(simbox.lengths == np.array([3,4,5])), "Problem with simbox.lengths"

if __name__ == '__main__':
    test_Simbox()
    test_LeesEdwards()
