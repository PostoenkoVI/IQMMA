#!/usr/bin/env python

'''
setup.py file for iqmma
'''
import os
from setuptools import setup, find_packages, Extension

version = open('VERSION').readline().strip()


setup(
    name                 = 'iqmma',
    version              = version,
    description          = '''A MS1 feature mapping for MS2 spectra identifications.''',
    long_description     = (''.join(open('README.md').readlines())),
    long_description_content_type = 'text/markdown',
    author               = 'Valeriy Postoenko & Leyla Garibova',
    author_email         = 'pyteomics@googlegroups.com',
    install_requires     = [line.strip() for line in open('requirements.txt')],
    classifiers          = ['Intended Audience :: Science/Research',
                            'Programming Language :: Python :: 3.9',
                            'Topic :: Education',
                            'Topic :: Scientific/Engineering :: Bio-Informatics',
                            'Topic :: Scientific/Engineering :: Chemistry',
                            'Topic :: Scientific/Engineering :: Physics'],
    license              = 'License :: OSI Approved :: Apache Software License',
    packages         = ['iqmma', ],
    package_data     = {'iqmma': ['default.ini', ]},
    entry_points         = {'console_scripts': ['iqmma = iqmma.iqmma:run',]},
    )
