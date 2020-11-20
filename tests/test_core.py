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


# =============================================================================
# TESTS
# =============================================================================
def test_obs_counter(carbognani2019):

    result = pyedra.obs_counter(carbognani2019, 8)

    expected = [85, 208, 306, 313, 338, 522]

    np.testing.assert_array_almost_equal(result, expected, 6)
