#!/usr/bin/env python

from os.path import exists
from setuptools import setup
#import versioneer

DISTNAME = 'seaobs'
PACKAGES = ['seaobs']
TESTS = [p + '.tests' for p in PACKAGES]
INSTALL_REQUIRES = ['numpy >= 1.7', 'scipy >=  0.18.0', 'xarray >= 0.8.2',
                    'dask >= 0.12.0', 'numba >=  0.30.0', ]
TESTS_REQUIRE = ['pytest >= 2.7.1']

URL = 'http://github.com/serazing/seaobs'
AUTHOR = 'Guillaume Serazin'
AUTHOR_EMAIL = 'guillaume.serazin@legos.obs-mip.fr'
LICENSE = 'Apache'
DESCRIPTION = 'Library to analyze ocean observations'

VERSION = '0.0.1'
#CMDCLASS = versioneer.get_cmdclass()

setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      url=URL,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      keywords='signal processing',
      packages=PACKAGES + TESTS,
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      install_requires=INSTALL_REQUIRES,
      zip_safe=False)
