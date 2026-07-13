def test_read_h5():
    import gamdpy as gp
    import h5py
    file = './tests/Data/LJ_cooled_0.70.h5'
    h5_a = gp.read_h5(file)
    h5_b = gp.tools.read_h5(file)
    h5_c = h5py.File(file, 'r')
    h5_d = gp.tools.TrajectoryIO(file).get_h5()
    assert h5_a == h5_b == h5_c == h5_d, "Outputs for reading h5 data are not the same"

if __name__ == "__main__":
    test_read_h5()
