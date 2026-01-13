py = import('python').find_installation(pure: false)
fs = import('fs')

# Build directory path
build_dir = meson.current_build_dir()
source_dir = meson.current_source_dir()

# Copy fortran files to build directory and compile them there
outflow_calc_build = custom_target(
  'outflow_calc_build',
  input: 'outflow_calc.f90',
  output: 'outflow_calc.stamp',
  command: [
    'sh', '-c',
    'cp @INPUT@ ' + build_dir + '/ && ' +
    'cd ' + build_dir + ' && ' +
    py.full_path() + ' -m numpy.f2py -c outflow_calc.f90 -m outflow_calc --lower && ' +
    'touch @OUTPUT@'
  ],
  install: false,
  build_by_default: true,
  console: true,
)

fast_tracer_build = custom_target(
  'fast_tracer_build',
  input: 'fast_tracer.f90',
  output: 'fast_tracer.stamp',
  command: [
    'sh', '-c',
    'cp @INPUT@ ' + build_dir + '/ && ' +
    'cd ' + build_dir + ' && ' +
    py.full_path() + ' -m numpy.f2py -c fast_tracer.f90 -m fast_tracer --lower && ' +
    'touch @OUTPUT@'
  ],
  install: false,
  build_by_default: true,
  console: true,
)

# Install the built .so files
meson.add_install_script(
  'sh', '-c',
  'find ' + build_dir + ' -name "outflow_calc*.so" -exec cp {} ' + 
  py.get_install_dir() + '/outflowpy/ \\;'
)

meson.add_install_script(
  'sh', '-c',
  'find ' + build_dir + ' -name "fast_tracer*.so" -exec cp {} ' +
  py.get_install_dir() + '/outflowpy/ \\;'
)
