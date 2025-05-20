""" Give information about the content of a .h5 file created by gamdpy
    Usage:

    info_h5 filename.h5
"""

import gamdpy as gp
import sys
import h5py

argv = sys.argv.copy()
argv.pop(0)  # remove scriptname
if __name__ == "__main__":
    if argv:
        filename = argv.pop(0) # get filename (.h5 added by script)
    else:
        filename = 'Data/LJ_r0.973_T0.70_toread.h5' # Used in testing
else:
    filename = 'Data/LJ_r0.973_T0.70_toread.h5' # Used in testing

print('Groups in', filename, ':')
with h5py.File(filename,'r') as f:
    for key in f.keys():
        print(key)
        if 'configuration' in key:
            grp = f[key]
            print('\tNumber of particles:', grp['vectors'].shape[1], ', Dimensions:', grp['vectors'].shape[2])
            print('\tsimbox_name:', grp.attrs['simbox_name'])
            print('\tsimbox_data:', grp.attrs['simbox_data'])
            if 'vectors' in grp.keys():
                print('\tvectors:', grp['vectors'].attrs['vector_columns'])
            if 'scalars' in grp.keys():
                print('\tscalars:', grp['scalars'].attrs['scalar_columns'])
            if 'topology' in grp.keys() and len(grp['topology'].keys())>0:
                print('\ttopology:')
                molecules = grp['topology']['molecules']
                for name in molecules.attrs['names']:
                    print(f"\t\t{name}, shape: {molecules[name].shape}")
                print(f"\t\tbonds: {grp['topology']['bonds'].shape}")
                print(f"\t\tangles: {grp['topology']['angles'].shape}")
                print(f"\t\tdihedrals: {grp['topology']['dihedrals'].shape}")
                
        elif key=='restarts':
            grp = f[key]
            print(f'\t{len(grp.keys())} restarts' )
        elif 'scalar_saver' in key:
            grp = f[key]
            print('\tscalar_names:', grp.attrs['scalar_names'])
            print('\tscalars, shape:', grp['scalars'].shape, ', dtype:', grp['scalars'].dtype)
        elif 'trajectory_saver' in key:
            grp = f[key]
            for subkey in grp.keys():
                print(f'\t{subkey}, shape: {grp[subkey].shape}, dtype: {grp[subkey].dtype}')
        else:
            print('\tUnrecognized')