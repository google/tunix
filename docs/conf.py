"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information # pylint: disable=g-line-too-long

project = "Tunix"
copyright = "2025, Tunix Developers"  # pylint: disable=redefined-builtin
author = "Tunix Developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration # pylint: disable=g-line-too-long

extensions = [
    "myst_nb",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.collections",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = [".rst", ".md", ".ipynb"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output # pylint: disable=g-line-too-long

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/img/tunix.png"

html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/google/tunix",
    "use_repository_button": True,  # add a "link to repository" button
    "navigation_with_keys": False,
}

# -- Options for sphinx-gallery ----------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "_collections/examples",  # path to your example scripts
    "gallery_dirs": (
        "_collections/gallery/"
    ),  # path to where to save gallery generated output
    "filename_pattern": "*.py",
}

# -- Options for myst -------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
nb_execution_mode = "off"
nb_execution_allow_errors = False
nb_render_image_options = {}
nb_execution_excludepatterns = [
    "*.ipynb",
]

# -- Options for sphinx-collections

collections = {
    "examples": {
        "driver": "copy_folder",
        "source": "../examples/",
        "ignore": "",
    }
}

suppress_warnings = ["misc.highlighting_failure"]
