import numpy as np
import gamdpy as gp

def test_colarray():
    size = (100,2)
    column_names = ('r', 'v', 'f')

    # setup a column array, and save it
    ca = gp.colarray(column_names=column_names, size=size)
    for column_name in column_names:
        ca[column_name] = np.random.uniform(size=size)
    ca.save('my_colarray')

    # load into new object, and check consistency
    ca2 = gp.colarray.load('my_colarray')
    for col in ca2.column_names:
        assert np.all(ca2[col]==ca[col])

    # check copy'ing
    ca3 = ca.copy()
    for col in ca3.column_names:
        assert np.all(ca3[col]==ca[col])

    # clean up
    gp.colarray.remove_files('my_colarray')

if __name__ == "__main__":  # pragma: no cover
    test_colarray()
