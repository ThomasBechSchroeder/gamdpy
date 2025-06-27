
def get_default_compute_flags() -> dict:
    """ Return dictionary with flags default compute flags.
    The boolean value determines whether the corresponding quantity is computed.
    """
    default_compute_flags = {'U':True, 'W':True, 'K': True, 'lapU':False,  'Fsq':False, 'stresses':False, 'Vol':False, 'Ptot':False}
    return default_compute_flags
