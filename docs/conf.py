# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Tunix'
copyright = '2025, Tunix Developers'
author = 'Tunix Developers'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.collections',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = [".rst", ".md", ".ipynb"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

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
    #"ignore_pattern": r"_test\.py",  # no gallery for test of examples
    #"doc_module": "optax",
    #"backreferences_dir": os.path.join("modules", "generated"),
}

# -- Options for myst -------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
nb_execution_mode = "force"
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
        "ignore": "BUILD"
    }
}

suppress_warnings = ['misc.highlighting_failure']
