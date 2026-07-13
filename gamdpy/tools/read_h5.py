import h5py

def read_h5(name, **kwargs) -> h5py.File:
    """ Read an h5 file. Shorthand for h5py.File(name, mode='r', **kwargs). """
    return h5py.File(name, mode='r', **kwargs)
