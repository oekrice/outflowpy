# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Outflowpy'
copyright = '2026, Oliver Rice, with some elements used from PFSSpy (David Stansby)'
author = 'Oliver Rice'

release = '0.0'
version = '0.0.7'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Add 'undocumented' functions
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Temporary debugging thing
import outflowpy
print("NAMESPACE:", [x for x in dir(outflowpy) if not x.startswith("_")])
