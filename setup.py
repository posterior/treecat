import sys

import numpy as np
from parsable import parsable
from setuptools import setup
from setuptools.extension import Extension

import eigency
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
except (IOError, ImportError, OSError) as e:
    sys.stderr.write('Failed to convert README.md to rst:\n  {}\n'.format(e))
    sys.stderr.flush()
    long_description = open('README.md').read()

for line in open('treecat/__init__.py'):
    if line.startswith('__version__ = '):
        version = line.strip().split()[-1][1:-1]

setup(
    name='pytreecat',
    version=version,
    description=(
        'A Bayesian latent tree model of multivariate multinomial data'),
    long_description=long_description,
    author='Fritz Obermeyer',
    author_email='fritz.obermeyer@gmail.com',
    url='https://github.com/posterior/treecat',
    packages=['treecat'],
    package_data={'treecat': 'testdata/*.csv'},
    ext_modules=cythonize(extensions),
    entry_points=parsable.find_entry_points('treecat'),
    install_requires=[
        'cython',
        'eigency',
        'jsonpickle',
        'matplotlib',
        'numpy',
        'pandas',
        'parsable',
        'scipy',
        'six',
    ],
    tests_require=[
        'flake8',
        'goftests',
        'matplotlib',
        'pytest',
    ],
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ])
