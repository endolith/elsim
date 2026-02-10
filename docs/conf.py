# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'elsim'
copyright = '2023, endolith'
author = 'endolith'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.mermaid',
    'myst_parser',
    'sphinx.ext.extlinks'
]

extlinks = {
    'doi': ('https://dx.doi.org/%s', 'doi:%s'),
    'wikipedia': ('https://en.wikipedia.org/wiki/%s', 'Wikipedia article: %s'),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'examples/README.md', 'examples/results']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_favicon = '_static/favicon.ico'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# So that examples/index gets image paths relative to that page (results/xxx not
# examples/results/xxx), we include from a copy of the README inside docs/examples/.
# Paths in that file (./results/) are then relative to the document and resolve
# correctly. Copy repo examples/README.md and examples/results/ into docs/examples/
# at build start so the doc and Sphinx can find them.
import os
import shutil


def prepare_examples_doc(app, config):
    srcdir = app.srcdir
    repo_examples = os.path.join(srcdir, '..', 'examples')
    docs_examples = os.path.join(srcdir, 'examples')
    for name, dest_subdir in (('README.md', ''), ('results', 'results')):
        src = os.path.join(repo_examples, name)
        if name == 'README.md':
            shutil.copy2(src, os.path.join(docs_examples, name))
        else:
            dest = os.path.join(docs_examples, dest_subdir)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(src, dest)


def setup(app):
    app.connect('config-inited', prepare_examples_doc)

# Add this to enable regular markdown mermaid syntax
myst_fence_as_directive = ["mermaid"]

# MyST configuration
myst_enable_extensions = [
    "colon_fence",    # For ::: fences
    "dollarmath",     # For $$
    "linkify",        # Auto-convert bare URLs to links
    "substitution",   # For {{ var }}
    "tasklist",       # For [ ] task lists
]

# Enable MyST to parse reST directives in markdown
myst_all_links_external = True
myst_heading_anchors = 3
myst_footnote_transition = True
myst_dmath_double_inline = True
myst_enable_checkboxes = True

# Add README.md as the index page
root_doc = 'index'  # or 'contents' in older versions
