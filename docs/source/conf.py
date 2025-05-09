# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

#   How to build the documentation:
# cd docs
# pip install sphinx myst_nb pydata-sphinx-theme
# make html
# make latexpdf

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
sys.path.insert(0, os.path.abspath('../..'))

import gamdpy

project = 'gamdpy'
author = 'Thomas B. Schrøder, Ulf R. Pedersen, Lorenzo Costigliola, Nicholas Bailey, Jesper Hansen and contributors'
release = gamdpy.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',  # enable markdown files (*.md), and Jupyter Notebooks (*.ipynb)
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

nb_execution_mode = "off"

templates_path = ['_templates']
exclude_patterns = []

# myst-nb configuration
nb_execution_timeout = -1
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
# html_logo = '_static/logo_777x147.png'