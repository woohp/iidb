from setuptools import setup, find_packages


setup(
    name='iidb',
    version='0.1',
    packages=find_packages(exclude=('tests',)),
    install_requires=['lmdb', 'zstandard'],
)
