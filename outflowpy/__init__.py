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
import outflowpy.plotting

from .input import Input
from .output import Output
from .pfss import pfss
from .outflow import outflow, outflow_fortran

import sys, types
sys.modules['sunpy.tests'] = types.ModuleType('sunpy.tests')
sys.modules['sunpy.tests.self_test'] = types.ModuleType('sunpy.tests.self_test')

__all__ = ['Input', 'Output', 'FortranTracer', 'PythonTracer', 'FastTracer', 'pfss', 'outflow', 'outflow_fortran', 'prepare_hmi_mid_crot', 'download_hmi_mdi_crot','prepare_hmi_mdi_time','outflow_calc', 'fast_tracer']


from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
