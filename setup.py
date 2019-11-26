#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='iidb',
    version='0.1',
    packages=find_packages(exclude=('tests',)),
    scripts=['bin/getimage', 'bin/merge-image-databases'],
    install_requires=['lmdb', 'zstandard>=0.12.0'],
)
