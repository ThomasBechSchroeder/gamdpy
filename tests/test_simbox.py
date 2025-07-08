def test_Orthorhombic():
    import pytest
    import types
    import gamdpy as gp
    import numpy as np

    ## Test init of Orthorhombic class
    with pytest.raises(ValueError, match="Length of lengths must be equal to D"):
        simbox = gp.Orthorhombic(D=3, lengths=np.array([3,4]))
    simbox = gp.Orthorhombic(D=3, lengths=np.array([3,4,5]))
    assert isinstance(simbox, gp.Orthorhombic), "Problem with Orthorhombic __init__"
    assert np.all(simbox.get_lengths() == np.array([3,4,5])), "Problem with Orthorhombic.get_lengths()"
    assert simbox.get_name()=="Orthorhombic", "Problem with Orthorhombic.get_name()"
    simbox.copy_to_device()
    simbox.copy_to_host()

    ## Test dist_sq_dr_function
    ri = np.array([2.5,3,4])
    rj = np.array([0.5,0.5,0.5])
    dr = 0*ri
    result = simbox.get_dist_sq_dr_function()(ri, rj, simbox.data_array, dr)
    assert result == 5.5, "Problem with Simbox.dist_sq_dr_function result"
    assert np.all( dr == np.array([-1, -1.5, -1.5]) ), "Problem with Orthorhombic.dist_sq_dr_function dr"

    ## Test dist_sq_function
    result = simbox.get_dist_sq_function()(ri, rj, simbox.data_array)
    assert result == 5.5, "Problem with Orthorhombic.dist_sq_function result"

    ## Test apply_PBC
    # Here the r is out of the box and PBC are applied by increasing img count
    r, img = np.array([4,5,6]), np.array([0,1,2])
    simbox.get_apply_PBC()(r, img, simbox.data_array)
    assert np.all( r   == np.array([1,1,1])), "Problem with Orthorhombic.apply_PBC:r positive img"
    assert np.all( img == np.array([1,2,3])), "Problem with Orthorhombic.apply_PBC:img positive img"
    # Here the r is inside the box and PBC are applied by decreasing img count
    r, img = -np.array([4,5,6]), np.array([0,1,2])
    simbox.get_apply_PBC()(r, img, simbox.data_array)
    assert np.all( r   == np.array([-1,-1,-1])), "Problem with Orthorhombic.apply_PBC:r negative img"
    assert np.all( img == np.array([-1,0,1])), "Problem with Orthorhombic.apply_PBC:img negative img"

    ## Test volume
    assert simbox.get_volume_function()(simbox.get_lengths())==3*4*5, "Problem with Orthorhombic.get_volume_function()"
    assert simbox.get_volume()==3.*4*5, "Problem with Orthorhombic.get_volume()"

    ## Test dist_moved_sq_function # This test is commented out because so is the function in the class
    #r_current = np.array([1,1,1])
    #r_last    = r_current + 0.75*simbox.get_lengths()
    #result = simbox.get_dist_moved_sq_function()(r_current, r_last, simbox.data_array, simbox.data_array)
    #assert result == 3.125, "Problem with Orthorhombic.dist_moved_sq_function"

    ## Test dist_moved_exceeds_limit_function
    r_current = np.array([1,1,1])
    r_last    = r_current + 0.50e0
    skin, cut = 2.e0, 1.e0
    result = simbox.get_dist_moved_exceeds_limit_function()(r_current, r_last, simbox.data_array, simbox.data_array, skin, cut)
    assert result == False, "Problem with Orthorhombic.dist_moved_exceeds_limit_function call 1"
    r_last    = r_current + 0.75e0
    result = simbox.get_dist_moved_exceeds_limit_function()(r_current, r_last, simbox.data_array, simbox.data_array, skin, cut)
    assert result == True, "Problem with Orthorhombic.dist_moved_exceeds_limit_function call 2"

    ## Test scale
    simbox.scale(scale_factor=0.1)
    assert np.all(simbox.data_array == 0.1*np.array([3,4,5], dtype=np.float32)), "Problem with Orthorhombic.scale"

    ## Test shift functions (used in SLLOD)
    assert simbox.get_loop_x_addition() == 0, "Problem with Orthorhombic.get_loop_x_addition: returns {simbox.get_loop_x_addition()} instead of 0"
    assert isinstance(simbox.get_loop_x_shift_function(), types.FunctionType), "Problem with Orthorhombic.get_loop_x_shift_function(): is not a python function"

def test_LeesEdwards():
    import pytest
    import types
    import gamdpy as gp
    import numpy as np

    # Test 1D error 
    # https://stackoverflow.com/questions/23337471/how-do-i-properly-assert-that-an-exception-gets-raised-in-pytest
    with pytest.raises(ValueError) as e_info:
        simbox = gp.LeesEdwards(D=1, lengths=np.array([3,4]), box_shift=1.0)
    assert e_info.type is ValueError

    # Test normal case
    simbox = gp.LeesEdwards(D=3, lengths=np.array([3,4,5]), box_shift=1.0)
    assert isinstance(simbox, gp.LeesEdwards), "Problem with LeesEdwards __init__"
    assert np.all(simbox.get_lengths() == np.array([3,4,5])), "Problem with LeesEdwards.get_lengths()"
    assert simbox.get_name()=="LeesEdwards", "Problem with LeesEdwards.get_name()"
    simbox.copy_to_device()
    simbox.copy_to_host()

    ## Test dist_sq_dr_function
    ri = np.array([1.5,0.5,0.5])
    rj = np.array([1.5,3.5,0.5])
    dr = 0*ri
    result = simbox.get_dist_sq_dr_function()(ri, rj, simbox.data_array, dr)
    assert result == 2., "Problem with LeesEdwards.dist_sq_dr_function result"
    assert np.all( dr == np.array([1., 1., 0.]) ), "Problem with LeesEdwards.dist_sq_dr_function dr"

    ## Test dist_sq_function
    result = simbox.get_dist_sq_function()(ri, rj, simbox.data_array)
    assert result == 2., "Problem with LeesEdwards.dist_sq_function result"

    ## Test apply_PBC
    # Here the r is out of the box and PBC are applied by increasing img count
    r, img = np.array([2.0,3.,6.]), np.array([0,1,2])
    simbox.get_apply_PBC()(r, img, simbox.data_array)
    assert np.all( r   == np.array([1.,-1.,1])), "Problem with LeesEdwards.apply_PBC:r positive img"
    assert np.all( img == np.array([0,2,3])), "Problem with LeesEdwards.apply_PBC:img positive img"

    ## Test volume
    assert simbox.get_volume_function()(simbox.get_lengths())==3*4*5, "Problem with LeesEdwards.get_volume_function()"
    assert simbox.get_volume()==3.*4*5, "Problem with LeesEdwards.get_volume()"

    ## Test dist_moved_exceeds_limit_function
    r_current = np.array([1,1,1])
    r_last    = r_current + 0.50e0
    last_data_array = simbox.data_array.copy()
    last_data_array[3] -= 0.1 # (assume boxshift has changed by 0.1)
    skin, cut = 2.e0, 1.e0
    result = simbox.get_dist_moved_exceeds_limit_function()(r_current, r_last, simbox.data_array, last_data_array, skin, cut)
    assert result == False, "Problem with LeesEdwards.dist_moved_exceeds_limit_function call 1"
    r_last    = r_current + 0.75e0
    result = simbox.get_dist_moved_exceeds_limit_function()(r_current, r_last, simbox.data_array, last_data_array, skin, cut)
    assert result == True, "Problem with LeesEdwards.dist_moved_exceeds_limit_function call 2"

    ## Test scale
    simbox.scale(scale_factor=0.1)
    assert np.all(simbox.data_array[:3] == 0.1*np.array([3,4,5], dtype=np.float32)), "Problem with LeesEdwards.scale"

    ## Test shift functions (used in SLLOD)
    assert simbox.get_loop_x_addition() == 1, "Problem with LeesEdwards.get_loop_x_addition: returns {simbox.get_loop_x_addition()} instead of 1"
    assert isinstance(simbox.get_loop_x_shift_function(), types.FunctionType), "Problem with LeesEdwards.get_loop_x_shift_function(): is not a python function"
    assert isinstance(simbox.get_update_box_shift(), types.FunctionType), "Problem with LeesEdwards.get_update_box_shift(): is not a python function"


if __name__ == '__main__':
    test_Orthorhombic()
    test_LeesEdwards()
