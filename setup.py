# Setup script for the pygridplot package
#
# Usage: python setup.py install
#
import os
from setuptools import setup, find_packages

DESCRIPTION = "pygridplot: Visualization tools for pygridgen"
LONG_DESCRIPTION = DESCRIPTION
NAME = "pygridplot"
VERSION = "0.0.1"
AUTHOR = "Lucas Nguyen (Geosyntec Consultants)"
AUTHOR_EMAIL = "lnguyen@geosyntec.com"
URL = "https://github.com/Geosyntec/pygridplot"
DOWNLOAD_URL = "https://github.com/Geosyntec/pygridplot/archive/master.zip"
LICENSE = "BSD 3-clause"
PACKAGES = find_packages()
PLATFORMS = "Python 2.7"
CLASSIFIERS = [
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Intended Audience :: Science/Research",
    "Topic :: Software Development :: Libraries :: Python Modules",
    'Programming Language :: Python :: 2.7',
]
INSTALL_REQUIRES = ['seaborn', 'numexpr', 'descartes', 'shapely', 'pyshp']
PACKAGE_DATA = {}

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    #data_files=DATA_FILES,
    platforms=PLATFORMS,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
)
