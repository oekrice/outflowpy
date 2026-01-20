py = import('python').find_installation(pure: false)

ext_suffix = py.get_variable('EXT_SUFFIX')
# Build directory path
build_dir = meson.current_build_dir()

is_windows = host_machine.system() == 'windows'

if is_windows
  # Windows uses .pyd extension
  ext_pattern = '*.pyd'
  shell = 'cmd'
  shell_arg = '/c'
else
  # Unix-like uses .so extension
  ext_pattern = '*.so'
  shell = 'sh'
  shell_arg = '-c'
endif

if is_windows
  outflow_calc_cmd = [
    'powershell', '-Command',
    'Copy-Item @INPUT@ ' + build_dir + '; ' +
    'Set-Location ' + build_dir + '; ' +
    '& ' + py.full_path() + ' -m numpy.f2py -c outflow_calc.f90 -m outflow_calc --lower; ' +
    'Copy-Item (Get-ChildItem -Filter "outflow_calc' + ext_pattern + '" -Recurse | Select-Object -First 1).FullName @OUTPUT@'
  ]
  
  fast_tracer_cmd = [
    'powershell', '-Command',
    'Copy-Item @INPUT@ ' + build_dir + '; ' +
    'Set-Location ' + build_dir + '; ' +
    '& ' + py.full_path() + ' -m numpy.f2py -c fast_tracer.f90 -m fast_tracer --lower; ' +
    'Copy-Item (Get-ChildItem -Filter "fast_tracer' + ext_pattern + '" -Recurse | Select-Object -First 1).FullName @OUTPUT@'
  ]
else
  outflow_calc_cmd = [
    'sh', '-c',
    'cp @INPUT@ ' + build_dir + '/ && ' +
    'cd ' + build_dir + ' && ' +
    py.full_path() + ' -m numpy.f2py -c outflow_calc.f90 -m outflow_calc --lower && ' +
    'cp `find . -name "outflow_calc*.so"` @OUTPUT@'
  ]
  
  fast_tracer_cmd = [
    'sh', '-c',
    'cp @INPUT@ ' + build_dir + '/ && ' +
    'cd ' + build_dir + ' && ' +
    py.full_path() + ' -m numpy.f2py -c fast_tracer.f90 -m fast_tracer --lower && ' +
    'cp `find . -name "fast_tracer*.so"` @OUTPUT@'
  ]
endif

# Build outflow_calc
outflow_calc_build = custom_target(
  'outflow_calc_build',
  input: 'outflow_calc.f90',
  output: 'outflow_calc' + ext_suffix,
  command: outflow_calc_cmd,
  install: true,
  install_dir: py.get_install_dir() / 'outflowpy',
  build_by_default: true,
  console: true,
)

# Build fast_tracer
fast_tracer_build = custom_target(
  'fast_tracer_build',
  input: 'fast_tracer.f90',
  output: 'fast_tracer' + ext_suffix,
  command: fast_tracer_cmd,
  install: true,
  install_dir: py.get_install_dir() / 'outflowpy',
  build_by_default: true,
  console: true,
)
