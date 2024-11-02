from setuptools import setup, Extension
import numpy

mod = Extension(
    'nufft.ext',
    sources=['./nufft/ext/main.cpp'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp", "-lfftw3"],
)

setup(
    name='nufft',
    ext_modules=[mod],
    packages=["nufft"],
)