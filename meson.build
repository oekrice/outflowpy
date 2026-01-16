project(
  'outflowpy',
  ['fortran','c'],
)

py = import('python').find_installation(pure: false)

py.install_sources(
  [
    'outflowpy/__init__.py',
    'outflowpy/analytic.py',
    'outflowpy/coords.py',
    'outflowpy/fieldline.py',
    'outflowpy/grid.py',
    'outflowpy/input.py',
    'outflowpy/interpolator.py',
    'outflowpy/map.py',
    'outflowpy/obtain_data.py',
    'outflowpy/outflow.py',
    'outflowpy/output.py',
    'outflowpy/pfss.py',
    'outflowpy/plotting.py',
    'outflowpy/sample_data.py',
    'outflowpy/tracing.py',
    'outflowpy/utils.py',
    'outflowpy/_version.py'
  ],
  subdir: 'outflowpy'
)

subdir('fortran')

# Install data files
py.install_sources(
  [
    'outflowpy/data/opt_flow.txt',
    # Add any other data files here
  ],
  subdir: 'outflowpy/data'
)

#Install tests
# Install tests
py.install_sources(
  [
    'outflowpy/tests/__init__.py',
    'outflowpy/tests/example_maps.py',
    'outflowpy/tests/test_analytic.py',
    'outflowpy/tests/test_coords.py',
    'outflowpy/tests/test_downloads.py',
    'outflowpy/tests/test_fieldline.py',
    'outflowpy/tests/test_outflow.py',
    'outflowpy/tests/test_pfss.py',
    'outflowpy/tests/test_tracers.py',
    'outflowpy/tests/test_utils.py',
  ],
  subdir: 'outflowpy/tests'
)

# Install test data if you have any
py.install_sources(
  [
    'outflowpy/tests/data/br_in.txt',
    'outflowpy/tests/data/br_out.txt',
    'outflowpy/tests/data/hmi_2210_smooth.txt',
    'outflowpy/tests/data/mdi_2000_smooth.txt',
    # Add any other test data files here
  ],
  subdir: 'outflowpy/tests/data',
  preserve_path: true
)
