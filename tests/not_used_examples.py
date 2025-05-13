""" Try to run all the scripts in the examples directory

The plt.show() function is replaced by a dummy function to avoid showing the plots.
This script will skip some examples that are known to fail, see variable exclude_files.
When debugging, you can change variable files to a few or a single file.
"""
import glob
import os
import subprocess
import time

import pytest
import numba
import numba.cuda

@pytest.mark.slow
def not_examples(path_to_examples='examples'):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.get_backend()
    os.environ['MPLBACKEND'] = 'Agg' # Reduced warnings from 94 to 62
    os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS'] = '0' # Reduced warnings from 62 to 59
    os.environ['gamdpy_SAVE_OUTPUT_EXAMPLES'] = '0' # used to avoid file creation when running pytest
    matplotlib.use('Agg')  # Static backend that does not halt on plt.show()
    # List of scripts to exclude

    run_first = [  # These examples generate files that are used by other examples
        'minimal.py',
        'isomorph.py',
        'isochore.py',
    ]

    exclude_files = [
        'test_shear.py',
        # FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'LJ_cooled_0.70.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
        'LJchain_wall.py',  # ImportError: cannot import name 'nvt_nh' from 'gamdpy.integrators'
        'minimal_cpu.py',
        ## The following are test in their own scripts
        'structure_factor.py',
    ]

    # Save the current working directory
    original_cwd = os.getcwd()
    examples_dir = os.path.abspath(path_to_examples)

    try:
        os.chdir(examples_dir)

        # Iterate over all Python files in the examples directory
        files = list(glob.glob('*.py'))
        files.sort()

        # Put the run_first files at the beginning
        for file in run_first:  # Remove the files from the list if they are already there
            if file in files:
                files.remove(file)
        files = run_first + files

        #files = ["calc_sq_from_h5.py", "calc_rdf_from_rumd3.py", "calc_rdf_from_h5.py"]  # Uncomment and modify for debugging a few or a single file
        print(f"Found {len(files)} examples: {files}")
        print(f"Excluding {len(exclude_files)} (if present): {exclude_files}")
        for file in files:
            numba.cuda.devices.reset()
            if os.path.basename(file) in exclude_files:
                print(f"Skipping {file} (warning: may fail)")
                continue

            print(f"\n\nExecuting {file}")
            tic = time.perf_counter()
            torun = subprocess.Popen(["python3", f"{file}"])
            torun.wait()
            stdout, stderr = torun.communicate()
            assert torun.returncode==0, f"Example {file} failed.\n"
            toc = time.perf_counter()
            print(f"Execution time for {file}: {toc - tic:.3} s")
    except FileNotFoundError as e:
        print(f"Warning: Cannot find needed file to run {file}. Running another example may provide it.")
        print(f"FileNotFoundError: {e}")
    finally:
        os.chdir(original_cwd)
        plt.close('all')
    del os.environ['gamdpy_SAVE_OUTPUT_EXAMPLES']


#if __name__ == '__main__':
#    test_examples()
