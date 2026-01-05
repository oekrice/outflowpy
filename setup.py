import subprocess
import os
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_py import build_py

class F2PyBuild(build_py):
    def run(self):

        import numpy

        build_dir = Path(self.build_lib) / "outflowpy"
        build_dir.mkdir(parents=True, exist_ok=True)
        # Attempt to compile the fotran code as per

        # Read OpenMP flags from environment
        openmp_flags = os.environ.get("OPENMP_FLAGS", "")

        # Compile outflow_calc.f90
        subprocess.check_call([
            "python", "-m", "numpy.f2py",
            "-c", "fortran/outflow_calc.f90",
            "-m", "outflow_calc"
        ] + openmp_flags.split())

        # Compile fast_tracer.f90
        subprocess.check_call([
            "python", "-m", "numpy.f2py",
            "-c", "fortran/fast_tracer.f90",
            "-m", "fast_tracer"
        ] + openmp_flags.split())

        # Move generated files into package directory
        for pattern in ["outflow_calc*.so", "outflow_calc*.pyd", "fast_tracer*.so", "fast_tracer*.pyd"]:
            for file in Path(".").glob(pattern):
                print(f"Moving {file} -> {build_dir / file.name}")
                file.rename(build_dir / file.name)

        super().run()

setup(
    name="outflowpy",
    version="0.1.0",
    description="Outflow field modelling with Fortran",
    author="Oliver Rice",
    author_email="oliverricesolar@gmail.com",
    packages=["outflowpy"],
    cmdclass={"build_py": F2PyBuild},
)
