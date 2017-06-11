from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='treecat',
    version='0.0.1',
    description='A tree-of-mixtures nonparametric Bayesian model',
    long_description=long_description,
    author='Fritz Obermeyer',
    author_email='fritz.obermeyer@gmail.com',
    packages=['treecat'],
    install_requires=['numpy', 'six'],
    extras_require={
        'tensorflow': ['tensorflow>=1.1.0'],
        'tensorflow with gpu': ['tensorflow-gpu>=1.1.0']
    },
    tests_require=['pytest', 'pytest-flake8'],
    license='Apache License 2.0')
