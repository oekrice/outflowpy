Installation
============

Outflowpy can be installed from PyPi using

```
pip install outflowpy
```

Due to the nature of the precompiled Fortran extensions, there are some limitations on the OS/Python versions required. Outflowpy will run for Python>=3.09 on linux and MacOs>=14.0, but requires Python>=3.13 on Windows.

Editable Installation
=====================

To install an editable version of the package for testing/development purposes, the following procedure is recommended (this may also allow installation if the above requirements are not met):

```
git clone https://github.com/oekrice/outflowpy.git
pip install -e . --no-build-isolation
```

Which should install all dependencies and install the Fortran extensions.

To run tests, from the base directory run:

```
pip install pytest
python -m pytest
```



