project(
  'outflowpy',
  version: '0.0.1'
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

py.install_sources(
  files(outflowpy + '/*.so'),
  subdir: 'outflowpy'
)
