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

import ez_setup

ez_setup.use_setuptools()

from setuptools import setup  # noqa

# =============================================================================
# FUNCTIONS
# =============================================================================

VERSION = "0.0.1"

setup(
    name="Pyedra",
    version=VERSION,
    description="Implementation of phase function for asteroids in Python",
    long_description=open("README.md").read(),
    author="Milagros Colazo",
    author_email="milirita.colazovinovo@gmail.com",
    url="https://github.com/milicolazo/Pyedra",
    py_modules=["ez_setup"],
    packages=["pyedra", "pyedra.datasets"],
    classifiers=["Programming Language :: Python :: 3.8"],
    license="The MIT License",
    install_requires=["numpy", "pandas", "scipy", "attr", "matplotlib"],
)
