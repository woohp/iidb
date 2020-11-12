#!/usr/bin/env python

from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext


__version__ = '0.2.0'
libraries = ['zstd', 'lz4', 'lmdb']  # add any libraries, such as sqlite3, here

ext_modules = [
    Pybind11Extension(
        'iidb', [
            'src/module.cpp',
        ],
        libraries=libraries,
        language='c++',
        define_macros=[('VERSION_INFO', __version__)],
    ),
]


setup(
    name='iidb',
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    scripts=['bin/getimage', 'bin/merge-image-databases'],
    test_suite='tests',
    zip_safe=False,
)
