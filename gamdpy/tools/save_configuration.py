

def save_configuration(configuration, filename:str, format="xyz", append=False):        
    """ Saves configuration to file.

    Helpful for, e.g., VMD visualization
    This current version only supports the standard xyz-format.

    Example
    -------

    >>> import os
    >>> import gamdpy as gp
    >>> import numpy as np
    >>> conf = gp.Configuration(D=3, N=10)
    >>> gp.tools.save_configuration(configuration=conf, filename="final.xyz", format="xyz")
    >>> os.remove("final.xyz")      # Removes file (for doctests)

    """

    import numpy as np
    append = "a" if append else "w"

    if format=="xyz":
        with open(filename, append) as f:
            np.savetxt(f, np.c_[configuration.ptype, configuration['r']],
                    header=f"{configuration.N}\n#Position xyz format - generated by gamdpy", comments="")
    else:
        raise ValueError("Format not supported")


