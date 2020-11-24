#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Pyedra Project (https://github.com/milicolazo/Pyedra/).
# Copyright (c) 2020, Milagros Colazo
# License: MIT
#   Full Text: https://github.com/milicolazo/Pyedra/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

import numpy as np

import pandas as pd

import pyedra.datasets

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================


@pytest.fixture(scope="session")
def carbognani2019():
    return pyedra.datasets.load_carbognani2019()


@pytest.fixture
def bad_data():
    return pd.DataFrame({"id": {0: 85}, "alpha": {0: 5}, "v": {0: 8}})

@pytest.fixture
def expected_filter():
    return pd.DataFrame({"id": {0: 85}, "H": {0: 7.492423}, "error_H": {0: 0.070257}, "G": {0: 0.0434}, "error_G": {0: 0.035114}, "R": {0: 0.991422}})
