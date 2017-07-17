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
    entry_points=parsable.find_entry_points('treecat'),
    install_requires=['jsonpickle', 'numpy', 'parsable', 'scipy', 'six'],
    tests_require=['flake8', 'goftests', 'matplotlib', 'pytest'],
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
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ])
