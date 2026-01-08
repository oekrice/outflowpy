project(
  'outflowpy',
  ['fortran'],
  version: '0.0.1'
)

py = import('python').find_installation()

# Build Fortran extension modules via f2py
py.extension_module(
  'outflow_calc',
  'fortran/outflow_calc.f90',
  install: true,
  subdir: 'outflowpy'
)

py.extension_module(
  'fast_tracer',
  'fortran/fast_tracer.f90',
  install: true,
  subdir: 'outflowpy'
)
