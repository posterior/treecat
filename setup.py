from parsable import parsable
from setuptools import setup

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
    entry_points=parsable.find_entry_points('treecat'),
    install_requires=['numpy', 'six', 'parsable'],
    extras_require={
        'tensorflow': ['tensorflow>=1.1.0'],
        'tensorflow with gpu': ['tensorflow-gpu>=1.1.0']
    },
    tests_require=['pytest', 'flake8', 'goftests'],
    license='Apache License 2.0')
