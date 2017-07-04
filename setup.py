import sys

from parsable import parsable
from setuptools import setup

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError, OSError) as e:
    sys.stderr.write('Failed to convert README.md to rst:\n  {}\n'.format(e))
    sys.stderr.flush()
    long_description = open('README.md').read()

version = open('treecat/version.py').read().strip().split()[-1][1:-1]

setup(
    name='tree-cat',
    version=version,
    description='A tree-of-mixtures nonparametric Bayesian model',
    long_description=long_description,
    author='Fritz Obermeyer',
    author_email='fritz.obermeyer@gmail.com',
    packages=['treecat'],
    entry_points=parsable.find_entry_points('treecat'),
    install_requires=['numpy', 'parsable', 'scipy', 'six'],
    tests_require=['pytest', 'flake8', 'goftests'],
    license='Apache License 2.0')
