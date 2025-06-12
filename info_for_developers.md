# Info for developers

## Implementing on GPU using numba.cuda

- Inherited from rumd3: pb (particles per block), tp (threads per particle)
- Hoping to avoid from rumd3: sorting (gets too complicated).

Synchronization is of the utmost importance for correctness. For example, all forces need to be calculated before the integrator starts moving the particles. 
Traditionally (and in rumd3) this is done by kernel-calls (a kernel is a function running on the GPU, called from the CPU): it is guaranteed that one kernel finishes before the next begins (unless you explicitly ask otherwise). 

Unfortunately, kernel calls are slow, especially in numba.cuda (as compared to c++.cuda). 
A rough estimate is that the maximum number of time steps per second (TPS) that can be achieved using kernel calls for synchronization is about 5000 - a far cry from the ~100.000 TPS that can be achieved for small systems using "grid synchronization": Calling 'grid.sync()' inside a kernel ensures all threads in the grid get synchronised (i.e., no threads proceed beyond this statement before all threads have reached this statement). 

There is a limit to how many thread blocks can be used with grid synchronization, which makes it inefficient at large system sizes, so we need to be able to choose between the two ways of synchronization. 
A good place to see how this is done without implementing all functions twice is in 'integrators.py'

## Todo / Issues

Open Todo's have been transfered to issues after developer meeting 4/6-25. For reference old todo-list is [here](old_todo.md).


## Various tools/strategies we will use
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- Git ( https://git-scm.com/doc, https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell ).
- Sphinx ( https://www.sphinx-doc.org/ ) for documentation, 
- ... to be hosted on readthedocs ( https://about.readthedocs.com/ ). Model: https://numba.readthedocs.io.
- Hypothesis (property-based testing, https://hypothesis.readthedocs.io ).
- doctest (no more failing examples in examples/docs!, see colarray.py for example).
- Jupyter notebooks for tutorials. Testing: nbmake?, testbook?
- Automatic testing upon uploading (CI). How to get access to GPU's?.
- Systematic benchmarking. Substantial degradation in performance will be considered a bug.

## Checklist for developing a new feature
- Copy code that resembles what you want to do and modify it to your needs.
- Write tests in a file placed in tests (run pytest to check that it works).
- Write an example and place it in examples, add it to the examples/README.md
- Write documentation in the docstrings of the code (run doctests to check that it works).
- Include the new feature in the documentation, e.g., you may need to edit docs/source/api.rst

## Some git cmd which might be useful

Getting hash of your master (Head)
```sh
git log --pretty=format:'%h' -n 1
```

Creating a public branch (on the repository) starting from the current master / branch
```sh
git checkout -b your_branch
git push -u origin your_branch
```

Difference in a single file between branches. Can use hash instead of master / branch
```sh
git diff origin branch -- gamdpy/Simulation.py
git diff hash1 hash2 -- gamdpy/Simulation.py
```
List the files that are different in two branches
```sh
git diff --name-only origin branch 
```
Show version of a file in another branch
```sh
git show branch:file
```

Reset the last commit. It will not delete any file but will go back removing the last commit and the adding related to that commit
```sh
git reset HEAD~
```

Reset all changes.
```sh
git reset HEAD --hard
```

## How to test the code
Running `pytest` in root (gamdpy) directory will run all tests.
This will use the settings in the file `pytest.ini`.

Install needed packages:

```sh
pip install pytest hypothesis scipy ipywidgets
```

Running pytest:

```sh
python3 -m pytest
```

Running all tests typically takes several minutes.
Slow tests can be skipped by running (test functions decorated with `@pytest.mark.slow`):

```sh
python3 -m pytest -m "not slow"
```

Running pytest with an -x option makes pytest stop after the first failure
```sh
pytest -x
```

Running pytest starting from the last failed test
```sh
pytest --lf
```

### Test of specific features

Test scripts are located in the `tests` directory. Most can be executed (in a verbose mode) as a script:

```bash
python3 tests/test_examples.py
```

Running doctest of a single file:

```bash
python3 -m doctest -v gamdpy/calculators/calculator_radial_distribution.py
```

### Coverage of tests

To see what part of the code is covered:

```sh
pip install coverage
coverage run -m pytest
```

After the tests are finished, do:

```sh
coverage report -m
```

or `coverage html`.

## Building documentation

To building the documentation using sphinx, https://www.sphinx-doc.org
(needs `pip install myst_nb pydata_sphinx_theme`)

Install the necessary packages:

```sh
pip install sphinx myst_nb pydata_sphinx_theme
```

Build documentation webpage:

```sh
cd docs
make html
```

Open a webpage with firefox (or your favorite browsers):

```sh
firefox build/html/index.html
```

Clean the build directory (optional):

```sh
make clean
```

## Make new release

1. Update version number in gamdpy/__init__.py and pyproject.toml
2. Test code
3. Build notebooks (tutorials): (coomandline)
4. Build documentation (se above)
5. push code to github
6. 'git tag  x.y.z'
7. git push --tags
8. Goto github and make release (button in right panel)
9. Add new version readthedocs (should be made automatic)
10. Update on pypi (requires twine and build)
    * rm dist/*
    * python3 -m build --sdist
    * python3 -m build --wheel
    * twine upload dist/*  (asks for token)


