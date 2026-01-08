from setuptools import setup, Extension
import numpy as np
import os 

extra_args = [] 
if os.environ.get("OUTFLOWPY_OPENMP") == "1": 
    extra_args.append("-fopenmp") 
    
Extension( 
    ... extra_compile_args=extra_args, 
    extra_link_args=extra_args, 
)

extensions = [
    Extension(
        name="outflowpy.outflow_calc",
        sources=["fortran/outflow_calc.f90"],
        extra_compile_args=extra_args,
        extra_link_args=extra_args,
    ),
    Extension(
        name="outflowpy.fast_tracer",
        sources=["fortran/fast_tracer.f90"],
        extra_compile_args=extra_args,
        extra_link_args=extra_args,
    ),
]

setup(
    name="outflowpy",
    version="0.0.1",
    packages=["outflowpy"],
    ext_modules=extensions,
)
