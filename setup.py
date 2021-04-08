#!/usr/bin/env python3
"""
pytoolsAL toolbox

usage: pip install -e . --user
OR sudo: sudo pip install -e .
"""
import os
import setuptools

filename = 'pytoolsAL/_version.py'
exec(compile(open(filename, 'rb').read(), filename, 'exec'), globals(), locals())

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='pytoolsAL',
    version=__version__,
    author='Anna Li',
    author_email='annajxli@gmail.com',
    description='misc personal Python utilities',
    long_description=long_description,
    url='https://github.com/annajxli/pytoolsAL',
    packages=['pytoolsAL'],
    package_dir={'pytoolsAL': 'pytoolsAL'},
    classifiers=[
        'Development Status :: 3 - Alpha',
    ],
    install_requires=[
        'numpy',
        'scikit-image',
        'setuptools',
        'scikit-learn',
        'imageio',
        'tifffile',
        'matplotlib',
        'ipywidgets']
)
