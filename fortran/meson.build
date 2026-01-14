py = import('python').find_installation(pure: false)

# Build directory path
build_dir = meson.current_build_dir()

# Build outflow_calc and copy .so as part of the build
outflow_calc_build = custom_target(
  'outflow_calc_build',
  input: 'outflow_calc.f90',
  output: 'outflow_fake.so',
  command: [
    'sh', '-c',
    'cp @INPUT@ ' + build_dir + '/ && ' +
    'cd ' + build_dir + ' && ' +
    py.full_path() + ' -m numpy.f2py -c outflow_calc.f90 -m outflow_calc --lower && ' +
    'cp $(find . -name "outflow_calc*.so" -print -quit) ' +
    '@MESON_BUILD_ROOT@/@OUTPUT@'   
  ],
  install: true,
  install_dir: py.get_install_dir() / 'outflowpy',
  build_by_default: true,
  console: true,
)

# Build fast_tracer and copy .so as part of the build
fast_tracer_build = custom_target(
  'fast_tracer_build',
  input: 'fast_tracer.f90',
  output: 'tracer_fake.so',
  command: [
    'sh', '-c',
    'cp @INPUT@ ' + build_dir + '/ && ' +
    'cd ' + build_dir + ' && ' +
    py.full_path() + ' -m numpy.f2py -c fast_tracer.f90 -m fast_tracer --lower && ' + 
    'cp $(find . -name "fast_tracer*.so" -print -quit) ' + 
    '@MESON_BUILD_ROOT@/@OUTPUT@'
  ],
  install: true,
  install_dir: py.get_install_dir() / 'outflowpy',
  build_by_default: true,
  console: true,
)
