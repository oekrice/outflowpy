py = import('python').find_installation(pure: false)
ext_suffix = py.get_variable('EXT_SUFFIX')
fs = import('fs')

# Build directory path
build_dir = meson.current_build_dir()

# Detect OS
is_windows = host_machine.system() == 'windows'

# Build outflow_calc
if is_windows
  outflow_calc_build = custom_target(
    'outflow_calc_build',
    input: 'outflow_calc.f90',
    output: 'outflow_calc' + ext_suffix,
    command: [
        'cmd', '/c',
        'if not exist "' + build_dir + '\fortran" mkdir "' + build_dir + '\fortran" && ' +
        'copy @INPUT@ "' + build_dir + '\fortran" && ' +
        'cd /d "' + build_dir + '\fortran" && ' +
        py.full_path() + ' -m numpy.f2py -c outflow_calc.f90 -m outflow_calc --lower && ' +
        'for %I in (outflow_calc*.pyd) do copy %I @OUTPUT@'
    ],
    install: true,
    install_dir: py.get_install_dir() / 'outflowpy',
    build_by_default: true,
    console: true,
  )
else
  outflow_calc_build = custom_target(
    'outflow_calc_build',
    input: 'outflow_calc.f90',
    output: 'outflow_calc' + ext_suffix,
    command: [
      'sh', '-c',
      'cp @INPUT@ ' + build_dir + '/ && ' +
      'cd ' + build_dir + ' && ' +
      py.full_path() + ' -m numpy.f2py -c outflow_calc.f90 -m outflow_calc --lower && ' +
      'cp `find . -name "outflow_calc*.so"` @OUTPUT@'
    ],
    install: true,
    install_dir: py.get_install_dir() / 'outflowpy',
    build_by_default: true,
    console: true,
  )
endif

# Build fast_tracer
if is_windows
  fast_tracer_build = custom_target(
    'fast_tracer_build',
    input: 'fast_tracer.f90',
    output: 'fast_tracer' + ext_suffix,
    command: [
        'cmd', '/c',
        'if not exist "' + build_dir + '\fortran" mkdir "' + build_dir + '\fortran" && ' +
        'copy @INPUT@ "' + build_dir + '\fortran" && ' +
        'cd /d "' + build_dir + '\fortran" && ' +
        py.full_path() + ' -m numpy.f2py -c fast_tracer.f90 -m fast_tracer --lower && ' +
        'for %I in (outflow_calc*.pyd) do copy %I @OUTPUT@'
    ],
    install: true,
    install_dir: py.get_install_dir() / 'outflowpy',
    build_by_default: true,
    console: true,
  )
else
  fast_tracer_build = custom_target(
    'fast_tracer_build',
    input: 'fast_tracer.f90',
    output: 'fast_tracer' + ext_suffix,
    command: [
      'sh', '-c',
      'cp @INPUT@ ' + build_dir + '/ && ' +
      'cd ' + build_dir + ' && ' +
      py.full_path() + ' -m numpy.f2py -c fast_tracer.f90 -m fast_tracer --lower && ' +
      'cp `find . -name "fast_tracer*.so"` @OUTPUT@'
    ],
    install: true,
    install_dir: py.get_install_dir() / 'outflowpy',
    build_by_default: true,
    console: true,
  )
endif
