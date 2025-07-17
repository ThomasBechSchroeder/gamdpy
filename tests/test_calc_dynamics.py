import gamdpy as gp
import numpy as np

# For (possibly) later: save (as pickle) the whole msd curve along with the others and then compare everything

def test_calc_dynamics():

    filename = 'examples/Data/LJ_r0.973_T0.70_toread.h5' # Used in testing
    # Load existing data
    output = gp.tools.TrajectoryIO(filename).get_h5()

    dynamics = gp.tools.calc_dynamics(output, 0, qvalues=7.5)
    
    max_msd = np.max(dynamics['msd'][:,0])
    assert np.isclose(0.04115, max_msd,rtol=1e-4), f'Maximum MSD should be 0.04115, it is {max_msd}'

def test_calc_dynamics_sllod():


    filename = 'tests/Data/sllod_data.h5' # Used in testing
    # Load existing data
    output = gp.tools.TrajectoryIO(filename).get_h5()

    dynamics = gp.tools.calc_dynamics(output, 0, qvalues=7.5)
    
    max_msd = np.max(dynamics['msd'][:,0])
    assert np.isclose(0.73197, max_msd,rtol=1e-4), f'Maximum MSD should be 0.73197, it is {max_msd}'

    

if __name__ == "__main__":
    test_calc_dynamics()
    test_calc_dynamics_sllod()
