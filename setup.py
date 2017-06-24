import numpy as np
import eigency
from parsable import parsable
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        'treecat.cTreecat',
        sources=['treecat/cTreecat.pyx', 'treecat/treecat.cpp'],
        extra_compile_args=['-std=c++11', '-O3', '-march=native'],
        include_dirs=(['.', np.get_include()] + eigency.get_includes()),
        extra_link_args=['-lm'],
        language='c++'),
]

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

setup(
    name='treecat',
    version='0.0.1',
    description='A tree-of-mixtures nonparametric Bayesian model',
    long_description=long_description,
    author='Fritz Obermeyer',
    author_email='fritz.obermeyer@gmail.com',
    packages=['treecat'],
    ext_modules=cythonize(extensions),
    entry_points=parsable.find_entry_points('treecat'),
    install_requires=['cython', 'eigency', 'numpy', 'six', 'parsable'],
    extras_require={
        'tensorflow': ['tensorflow>=1.1.0'],
        'tensorflow with gpu': ['tensorflow-gpu>=1.1.0']
    },
    tests_require=['pytest', 'flake8', 'goftests'],
    license='Apache License 2.0')
