from parsable import parsable
from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='treetree',
    version='0.0.1',
    description='A tree-of-trees nonparametric Bayesian model',
    long_description=long_description,
    author='Fritz Obermeyer',
    author_email='fritz.obermeyer@gmail.com',
    packages=['treetree'],
    install_requires=['pandas', 'numpy', 'six'],
    extras_require={'tensorflow': ['tensorflow>=1.1.0'],
                    'tensorflow with gpu': ['tensorflow-gpu>=1.1.0']},
    tests_require=['pytest', 'pytest-pep8', 'flake8'],
    license='Apache License 2.0',
    entry_points=parsable.find_entry_points('treetree'),
)
