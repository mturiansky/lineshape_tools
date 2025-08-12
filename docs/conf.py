# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

try:
    import lineshape_tools
except ImportError:
    sys.path.insert(0, os.path.abspath("../src"))
    import lineshape_tools

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "lineshape_tools"
copyright = "2025, Mark E. Turiansky"  # noqa: A001
author = "Mark E. Turiansky"

release = lineshape_tools.__version__
version = lineshape_tools.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoapi_dirs = ["../src/lineshape_tools"]
autoapi_type = ["python"]
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

python_maximum_signature_line_length = 99


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
