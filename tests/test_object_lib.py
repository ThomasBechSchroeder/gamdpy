def test_object_lib():
    import gamdpy as gp
    import numpy as np
    ''' Test for the object_lib.py file. object_lib.py contains examples of Configuration and PairPotential classes '''

    # Test configuration_SC
    from object_lib import configuration_SC
    assert isinstance(configuration_SC, gp.Configuration), "Problem with configuration_SC object type"
    assert np.unique(configuration_SC.ptype)==0, "Problem with configuration_SC.ptypes"
    assert configuration_SC.N == 1000, "Problem with configuration_SC.N"
    assert configuration_SC.D == 3, "Problem with configuration_SC.D"
    assert configuration_SC.compute_flags == {'U': True, 'W': True, 'K': True, 'lapU': False, 'Fsq': False, 'stresses': False, 'Vol': False, 'Ptot': False}, "Problem with configuration_SC.compute_flags"

    # Test pairpot_LJ
    from object_lib import pairpot_LJ
    assert isinstance(pairpot_LJ, gp.PairPotential), "Problem with pairpot_LJ object type"
    assert pairpot_LJ.params_user == [1.0, 1.0, 2.5], "Problem with pairpot_LJ.params_user"
    assert pairpot_LJ.exclusions == None, "Problem with pairpot_LJ.exclusions"
    assert pairpot_LJ.max_num_nbs == 1000, "Problem with pairpot_LJ.max_num_nbs"

if __name__ == '__main__':
    test_object_lib()
