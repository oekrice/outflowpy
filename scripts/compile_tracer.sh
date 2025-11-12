#!/bin/bash
python -m numpy.f2py -c ../fortran/fast_tracer.f90 -m fast_tracer
mv fast_tracer.cpython-313-x86_64-linux-gnu.so ../outflowpy
python 3_image_optimisation.py
