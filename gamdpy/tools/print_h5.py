import h5py

def print_h5_structure(node, indent=0):
    """ Recursively print groups and datasets with metadata of an h5 file.

    Example
    -------

    >>> import gamdpy as gp
    >>> sim = gp.get_default_sim()
    >>> for _ in sim.run_timeblocks(): pass
    >>> gp.tools.print_h5_structure(sim.output)
    initial_configuration/ (Group)
        ptype  (Dataset, shape=(2048,), dtype=int32)
        r_im  (Dataset, shape=(2048, 3), dtype=int32)
        scalars  (Dataset, shape=(2048, 4), dtype=float32)
        topology/ (Group)
            angles  (Dataset, shape=(0,), dtype=int32)
            bonds  (Dataset, shape=(0,), dtype=int32)
            dihedrals  (Dataset, shape=(0,), dtype=int32)
            molecules/ (Group)
        vectors  (Dataset, shape=(3, 2048, 3), dtype=float32)
    scalar_saver/ (Group)
        scalars  (Dataset, shape=(8, 64, 3), dtype=float32)
    trajectory_saver/ (Group)
        images  (Dataset, shape=(8, 12, 2048, 3), dtype=int32)
        positions  (Dataset, shape=(8, 12, 2048, 3), dtype=float32)

    """
    for key, item in node.items():
        pad = "    " * indent
        if isinstance(item, h5py.Dataset):
            print(f"{pad}{key}  (Dataset, shape={item.shape}, dtype={item.dtype})")
        elif isinstance(item, h5py.Group):
            print(f"{pad}{key}/ (Group)")
            print_h5_structure(item, indent+1)
        else:  # This should not be relevant
            print(f"{pad}{key}  (Unknown type: {type(item)})")


def print_h5_attributes(obj, path="/"):
    """ Recursively print attrs of every group/dataset of an h5 file.

    Example
    -------

    >>> import gamdpy as gp
    >>> sim = gp.get_default_sim()
    >>> for _ in sim.run_timeblocks(): pass
    >>> gp.tools.print_h5_attributes(sim.output)
    Attributes at /:
        - dt: 0.005
        - script_content: ...
        - script_name: ...
    Attributes at /initial_configuration/:
        - simbox_data: [12.815602 12.815602 12.815602]
        - simbox_name: Orthorhombic
    Attributes at /initial_configuration/scalars:
        - scalar_columns: ['U' 'W' 'K' 'm']
    Attributes at /initial_configuration/topology/molecules/:
        - names: []
    Attributes at /initial_configuration/vectors:
        - vector_columns: ['r' 'v' 'f']
    Attributes at /scalar_saver/:
        - compression_info: gzip with opts 4
        - scalar_names: ['U' 'W' 'K']
        - steps_between_output: 16
    Attributes at /trajectory_saver/:
        - compression_info: gzip with opts 4

    """
    # obj could be the File or a Group
    if obj.attrs:
        print(f"Attributes at {path}:")
        for name, val in obj.attrs.items():
            if name == 'script_content' or name == 'script_name':  # Exclude since output is unpredictable (and untestable)
                print(f'    - {name}: ...')
            else:
                print(f"    - {name}: {val}")
    # Recurse into sub‐groups/datasets
    if isinstance(obj, h5py.Group):
        for key, sub in obj.items():
            print_h5_attributes(sub, path + key + ("/" if isinstance(sub, h5py.Group) else ""))
