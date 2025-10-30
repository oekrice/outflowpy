# Import pfsspy sub-modules to have them available through pfsspy.{name}
try:
    import outflowpy.analytic
except ModuleNotFoundError:
    # If sympy isn't installed
    pass
import outflowpy.coords
import outflowpy.fieldline
# Import this to register map sources
import outflowpy.map
import outflowpy.sample_data
import outflowpy.tracing
import outflowpy.utils
import outflowpy.obtain_data

from .input import Input
from .output import Output
from .pfss import pfss
from .outflow import outflow
from .outflow import outflow_fortran

__all__ = ['Input', 'Output', 'pfss', 'outflow', 'outflow_calc']


from ._version import get_versions

__version__ = get_versions()['version']
del get_versions


__citation__ = __bibtex__ = """
@article{Stansby2020,
doi = {10.21105/joss.02732},
url = {https://doi.org/10.21105/joss.02732},
year = {2020},
publisher = {The Open Journal},
volume = {5},
number = {54},
pages = {2732},
author = {David Stansby and Anthony Yeates and Samuel T. Badman},
title = {pfsspy: A Python package for potential field source surface modelling},
journal = {Journal of Open Source Software}
}
"""
