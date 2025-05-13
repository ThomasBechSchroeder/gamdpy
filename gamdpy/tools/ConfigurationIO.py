# This class is used for loading and saving configurations 
# Can be also used to convert between output formats (rumd3 -> gamdpy supported so far)
import sys
import numpy as np
import os
from gamdpy import Configuration

class ConfigurationIO():
    """
    This class handles loading and saving of gumdpy configuration object.
    The default saving is as .h5 but it's possible to convert to different formats.

    Paramters
    ---------

    configuration : Configuration
        Configuration object to save

    output_name :
        Name of the output file

    output_format : str
        String describing the type of outfile desired

    Examples
    --------
    """
    
    import h5py

    def __init__(self, configuration: Configuration, output_name: str, output_format: str, input_name: str):
        return

    def __save__(self):
        return

    def __load__(self):
        return

    def save_as_h5(self):
        return

    def save_as_xyz(self):
        return

    def save_as_lammps(self):
        return

    def load_from_h5(self):
        return

    def load_from_xyz(self):
        return

    def load_from_lammps(self):
        return
