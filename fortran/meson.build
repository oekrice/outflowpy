py = import('python').find_installation(pure: false)

# Use numpy's f2py meson backend
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

# Get numpy include directory
incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

# Get f2py include directory  
inc_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
fortranobject_c = inc_f2py / 'fortranobject.c'

# Generate f2py wrappers for outflow_calc
outflow_calc_source = custom_target('outflow_calcmodule.c',
  input : ['outflow_calc.f90'],
  output : ['outflow_calcmodule.c', 'outflow_calc-f2pywrappers2.f90'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'outflow_calc', '--lower']
)

# Build outflow_calc extension
py.extension_module('outflow_calc',
  ['outflow_calc.f90', outflow_calc_source, fortranobject_c],
  incdir_numpy,
  c_args: numpy_nodepr_api,
  dependencies : py.dependency(),
  install : true,
  subdir: 'outflowpy'
)

# Generate f2py wrappers for fast_tracer
fast_tracer_source = custom_target('fast_tracermodule.c',
  input : ['fast_tracer.f90'],
  output : ['fast_tracermodule.c', 'fast_tracer-f2pywrappers2.f90'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'fast_tracer', '--lower']
)

# Build fast_tracer extension
py.extension_module('fast_tracer',
  ['fast_tracer.f90', fast_tracer_source, fortranobject_c],
  incdir_numpy,
  c_args: numpy_nodepr_api,
  dependencies : py.dependency(),
  install : true,
  subdir: 'outflowpy'
)
