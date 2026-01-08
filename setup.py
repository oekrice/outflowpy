from setuptools import setup, Extension
import numpy as np

extensions = [
    Extension(
        name="outflowpy.outflow_calc",
        sources=["fortran/outflow_calc.f90"],
    ),
    Extension(
        name="outflowpy.fast_tracer",
        sources=["fortran/fast_tracer.f90"],
    ),
]

setup(
    name="outflowpy",
    version="0.0.1",
    packages=["outflowpy"],
    ext_modules=extensions,
)
