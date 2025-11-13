import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_py import build_py

class F2PyBuild(build_py):
    def run(self):

        import numpy

        build_dir = Path(self.build_lib) / "outflowpy"
        build_dir.mkdir(parents=True, exist_ok=True)
        # Attempt to compile the fotran code as per
        subprocess.check_call([
            "python", "-m", "numpy.f2py",
            "-c", "fortran/outflow_calc.f90",
            "-m", "outflow_calc"
        ])

        subprocess.check_call([
            "python", "-m", "numpy.f2py",
            "-c", "fortran/fast_tracer.f90",
            "-m", "fast_tracer", "--f90flags='-fopenmp'", "-lgomp"
        ])

        for file in Path(".").glob("outflow_calc*.so"):
            print('File found and moving', build_dir / file.name)
            file.rename(build_dir / file.name)
        for file in Path(".").glob("outflow_calc*.pyd"):
            file.rename(build_dir / file.name)
        for file in Path(".").glob("outflow_calc*.c"):
            file.rename(build_dir / file.name)

        for file in Path(".").glob("fast_tracer*.so"):
            print('File found and moving', build_dir / file.name)
            file.rename(build_dir / file.name)
        for file in Path(".").glob("fast_tracer*.pyd"):
            file.rename(build_dir / file.name)
        for file in Path(".").glob("fast_tracer*.c"):
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
