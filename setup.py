from setuptools import setup, Extension
import numpy

inc_alglib = "/usr/include/libalglib/"

setup(
    name='NUFFT',
    version='0.1',
    ext_modules=[
        Extension(
            'nufft',
            sources=['./main.cpp'],
            include_dirs=[numpy.get_include(), inc_alglib],
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp", "-lfftw3"],
        )
    ],
)