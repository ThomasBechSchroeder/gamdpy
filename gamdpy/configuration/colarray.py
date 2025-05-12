import numpy as np

# Can be doctested: https://docs.python.org/3/library/doctest.html
# python3 -m doctest    colarray.py  # No output == No problem!
# python3 -m doctest -v colarray.py  # The verbose version

class colarray():
    """ The Column array Class

    A class storing several sets ('columns') of lengths with identical dimensions in a single numpy array. Strings are used as indices along the zeroth dimension corresponding to different columns of lengths.
    
    Examples
    --------

    Storage for positions, velocities, and forces, for 1000 particles in 2 dimensions:

    >>> ca = colarray(('r', 'v', 'f'), size=(1000,2))
    >>> ca.shape
    (3, 1000, 2)
    >>> ca.column_names
    ('r', 'v', 'f')
    >>> # Data is accessed via string indices (similar to dataframes in pandas):
    >>> ca['r'] = np.ones((1000,2))
    >>> ca['v'] = 2   # Broadcast by numpy to correct shape
    >>> print(ca['r'] + 0.01*ca['v'])
    [[1.02 1.02]
     [1.02 1.02]
     [1.02 1.02]
     ...
     [1.02 1.02]
     [1.02 1.02]
     [1.02 1.02]]
    """
    
    # Most error handling is left to be handled by numpy, as it gives usefull error messages 
    # (illustrated in the documentation string above).
    
    def __init__(self, column_names, size, dtype=np.float32, array=None):
        self.column_names = column_names
        self.dtype = dtype
        self.indices = {key:index for index,key in enumerate(column_names)}
        if type(array)==np.ndarray: # Used, e.g.,  when loading from file
            self.array = array
        else:
            self.array = np.zeros((len(column_names), *size), dtype=dtype) 
            
        self.shape = self.array.shape

    def __setitem__(self, key, data):
        self.array[self.indices[key]] = data
        
    def __getitem__(self, key):
        return self.array[self.indices[key]]
   
    def __repr__(self):
        return 'colarray('+str(tuple(self.indices.keys()))+', '+self.array.shape[1:].__repr__()+')\n'+self.array.__repr__()
    
    def copy(self):
        return colarray(self.column_names, self.shape, self.dtype, self.array.copy())


    def save(self, filename):
        """
        Save a colarray to disk.
        >>> ca = colarray(('r', 'v', 'f'), size=(1000,2))
        >>> ca.save('my_colarray')
        >>> # Remove the files used for storage of the colarray:
        >>> colarray.remove_files('my_colarray')
        """
        np.save(f'{filename}.npy', self.array)
        with open(f'{filename}.col', 'w') as f:    # Use pickle / json
            f.write(str(len(self.column_names)) + '\n')
            for key in self.column_names:
                f.write(key + '\n')
        return


    def load(filename):
        """
        Load a colarray from disk.
        >>> ca = colarray(('r', 'v', 'f'), size=(1000,2))
        >>> ca['f'] = np.random.uniform(size=(1000,2))    
        >>> ca.save('my_colarray')
        >>> ca2 = colarray.load('my_colarray')
        >>> for col in ca.column_names:
        ...     print(np.all(ca2[col]==ca[col]))
        True
        True
        True

        Remove the files used for storage of the colarray:
        >>> colarray.remove_files('my_colarray')
        
        The file(s) needs to be present:
        >>> ca2 = colarray.load('my_colarray')
        Traceback (most recent call last):
            ...
        FileNotFoundError: [Errno 2] No such file or directory: 'my_colarray.col'
        """
        
        with open(f'{filename}.col', 'r') as f:
            num_columns = int(f.readline())
            column_names = []
            for i in range(num_columns):
                column_names.append(f.readline()[:-1]) # removing '\n'
        array = np.load(f'{filename}.npy')
        return colarray(column_names, array.shape[1:], array=array)
    
    def remove_files(filename):
        """
        Remove files storing a colarray
        >>> ca = colarray(('r', 'v', 'f'), size=(1000,2))
        >>> ca.save('my_colarray')
        >>> colarray.remove_files('my_colarray')
        """
        
        import os
        os.remove(f'{filename}.col') 
        os.remove(f'{filename}.npy')

