#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Pyedra Project (https://github.com/milicolazo/Pyedra/).
# Copyright (c) 2020, Milagros Colazo
# License: MIT
#   Full Text: https://github.com/milicolazo/Pyedra/blob/master/LICENSE

# =====================================================================
# DOCS
# =====================================================================

"""This file is for distribute and install Pyedra"""

# ======================================================================
# IMPORTS
# ======================================================================

import os
import pathlib

import ez_setup

ez_setup.use_setuptools()

from setuptools import setup  # noqa

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


REQUIREMENTS = ["numpy", "pandas", "scipy", "attrs", "matplotlib"]

with open(PATH / "pyedra" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break


with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()


# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="Pyedra",
    version=VERSION,
    description="Implementation of phase function for asteroids in Python",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Milagros Colazo",
    author_email="milirita.colazovinovo@gmail.com",
    url="https://github.com/milicolazo/Pyedra",
    py_modules=["ez_setup"],
    packages=[
        "pyedra",
        "pyedra.datasets",
    ],
    license="The MIT License",
    install_requires=REQUIREMENTS,
    keywords=["pyedra", "asteroid", "phase function"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
)
