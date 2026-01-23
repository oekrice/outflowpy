py = import('python').find_installation(pure: false)
ext_suffix = py.get_variable('EXT_SUFFIX')
build_dir = meson.current_build_dir()
is_windows = host_machine.system() == 'windows'

if is_windows
  outflow_calc_cmd = [
    'powershell', '-Command',
    'Copy-Item @INPUT@ ' + build_dir + '; ' +
    'Set-Location ' + build_dir + '; ' +
    '$env:F2PY_NO_SHARED="1"; ' +
    '& ' + py.full_path() + ' -m numpy.f2py -c outflow_calc.f90 -m outflow_calc --lower; ' +
    '$pyd = Get-ChildItem -Filter "*outflow_calc*.pyd" -Recurse | Select-Object -First 1; ' +
    'if (-not $pyd) { throw "Could not find outflow_calc .pyd file" }; ' +
    # Find ALL DLLs in the build directory
    '$dlls = Get-ChildItem -Filter "*.dll" -Recurse; ' +
    # Copy all DLLs to the package directory
    'foreach ($dll in $dlls) { Copy-Item $dll.FullName ($pyd.Directory.FullName) -Force }; ' +
    'Copy-Item $pyd.FullName @OUTPUT@; ' +
    # Also copy DLLs to output directory for packaging
    'foreach ($dll in $dlls) { Copy-Item $dll.FullName (Split-Path @OUTPUT@) -Force }'
  ]
  
  fast_tracer_cmd = [
    'powershell', '-Command',
    'Copy-Item @INPUT@ ' + build_dir + '; ' +
    'Set-Location ' + build_dir + '; ' +
    '$env:F2PY_NO_SHARED="1"; ' +
    '& ' + py.full_path() + ' -m numpy.f2py -c fast_tracer.f90 -m fast_tracer --lower; ' +
    '$pyd = Get-ChildItem -Filter "*fast_tracer*.pyd" -Recurse | Select-Object -First 1; ' +
    'if (-not $pyd) { throw "Could not find fast_tracer .pyd file" }; ' +
    # Find ALL DLLs in the build directory
    '$dlls = Get-ChildItem -Filter "*.dll" -Recurse; ' +
    # Copy all DLLs to the package directory
    'foreach ($dll in $dlls) { Copy-Item $dll.FullName ($pyd.Directory.FullName) -Force }; ' +
    'Copy-Item $pyd.FullName @OUTPUT@; ' +
    # Also copy DLLs to output directory for packaging
    'foreach ($dll in $dlls) { Copy-Item $dll.FullName (Split-Path @OUTPUT@) -Force }'
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
