"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Tunix"
copyright = "2025, Tunix Developers"  # pylint: disable=redefined-builtin
author = "Tunix Developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.collections",
    # api docs
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_collections/examples/model_load/from_safetensor_load/*",
]

source_suffix = [".rst", ".md", ".ipynb"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

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
        "ignore": "../examples/model_load/from_safetensor_load/*",
    }
}


suppress_warnings = ["misc.highlighting_failure"]


# -- Options for the API reference

default_role = "py:obj"

napoleon_include_init_with_doc = False

autodoc_default_options = {
    "members": True,
    "imported-members": True,
    "undoc-members": True,
}


intersphinx_mapping = {
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
