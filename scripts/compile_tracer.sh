#!/bin/bash
python -m numpy.f2py -c ../fortran/fast_tracer.f90 -m fast_tracer --f90flags="-fopenmp" -lgomp
mv fast_tracer.cpython-313-x86_64-linux-gnu.so ../outflowpy
python 2a_plot_date.py

