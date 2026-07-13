import h5py

def print_h5_structure(node, indent=0):
    """ Recursively print groups and datasets with metadata of an h5 file.

    Example
    -------

    >>> import gamdpy as gp
    >>> sim = gp.get_default_sim(num_timeblocks=2)
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
    restarts/ (Group)
        restart0000/ (Group)
            integrator_state  (Dataset, shape=(1,), dtype=float32)
            ptype  (Dataset, shape=(2048,), dtype=int32)
            r_im  (Dataset, shape=(2048, 3), dtype=int32)
            scalars  (Dataset, shape=(2048, 4), dtype=float32)
            topology/ (Group)
                angles  (Dataset, shape=(0,), dtype=int32)
                bonds  (Dataset, shape=(0,), dtype=int32)
                dihedrals  (Dataset, shape=(0,), dtype=int32)
                molecules/ (Group)
            vectors  (Dataset, shape=(3, 2048, 3), dtype=float32)
        restart0001/ (Group)
            integrator_state  (Dataset, shape=(1,), dtype=float32)
            ptype  (Dataset, shape=(2048,), dtype=int32)
            r_im  (Dataset, shape=(2048, 3), dtype=int32)
            scalars  (Dataset, shape=(2048, 4), dtype=float32)
            topology/ (Group)
                angles  (Dataset, shape=(0,), dtype=int32)
                bonds  (Dataset, shape=(0,), dtype=int32)
                dihedrals  (Dataset, shape=(0,), dtype=int32)
                molecules/ (Group)
            vectors  (Dataset, shape=(3, 2048, 3), dtype=float32)
    scalars/ (Group)
        scalars  (Dataset, shape=(2, 64, 3), dtype=float32)
        steps  (Dataset, shape=(65,), dtype=int32)
    trajectory/ (Group)
        images  (Dataset, shape=(2, 12, 2048, 3), dtype=int32)
        positions  (Dataset, shape=(2, 12, 2048, 3), dtype=float32)
        ptypes  (Dataset, shape=(2, 12, 2048), dtype=int32)
        steps  (Dataset, shape=(12,), dtype=int32)
        topologies/ (Group)
            block0000/ (Group)
                angles  (Dataset, shape=(0,), dtype=int32)
                bonds  (Dataset, shape=(0,), dtype=int32)
                dihedrals  (Dataset, shape=(0,), dtype=int32)
                molecules/ (Group)
            block0001/ (Group)
                angles  (Dataset, shape=(0,), dtype=int32)
                bonds  (Dataset, shape=(0,), dtype=int32)
                dihedrals  (Dataset, shape=(0,), dtype=int32)
                molecules/ (Group)

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
    >>> sim = gp.get_default_sim(num_timeblocks=2)
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
    Attributes at /restarts/:
        - timeblocks_between_restarts: 1
    Attributes at /restarts/restart0000/:
        - simbox_data: [12.815602 12.815602 12.815602]
        - simbox_name: Orthorhombic
    Attributes at /restarts/restart0000/scalars:
        - scalar_columns: ['U' 'W' 'K' 'm']
    Attributes at /restarts/restart0000/topology/molecules/:
        - names: []
    Attributes at /restarts/restart0000/vectors:
        - vector_columns: ['r' 'v' 'f']
    Attributes at /restarts/restart0001/:
        - simbox_data: [12.815602 12.815602 12.815602]
        - simbox_name: Orthorhombic
    Attributes at /restarts/restart0001/scalars:
        - scalar_columns: ['U' 'W' 'K' 'm']
    Attributes at /restarts/restart0001/topology/molecules/:
        - names: []
    Attributes at /restarts/restart0001/vectors:
        - vector_columns: ['r' 'v' 'f']
    Attributes at /scalars/:
        - compression_info: gzip with opts 4
        - scalar_columns: ['U' 'W' 'K']
        - scheduler: Lin
        - scheduler_info: {"steps_between": 16, "npoints": null}
        - steps_between_output: 16
    Attributes at /trajectory/:
        - compression_info: gzip with opts 4
        - num_timeblocks: 2
        - scheduler: Log2
        - scheduler_info: {}
        - steps_per_timeblock: 1024
        - trajectory_columns: ['r' 'img']
        - update_ptype: False
        - update_topology: False
    Attributes at /trajectory/topologies/block0000/molecules/:
        - names: []
    Attributes at /trajectory/topologies/block0001/molecules/:
        - names: []

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
