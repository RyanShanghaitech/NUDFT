from setuptools import setup, Extension
import numpy

mod = Extension(
    'nudft.ext',
    sources=['./nudft/ext/main.cpp'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    name='nudft',
    ext_modules=[mod],
    packages=["nudft"],
)