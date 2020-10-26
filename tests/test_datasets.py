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

# =============================================================================
# TESTS
# =============================================================================


def test_load_carbognani2019():

    result = pyedra.datasets.load_carbognani2019()

    assert isinstance(result, pd.DataFrame)
    np.testing.assert_almost_equal(result["id"].mean(), 283.042553, 6)
    np.testing.assert_almost_equal(result["alpha"].mean(), 9.228085, 6)
    np.testing.assert_almost_equal(result["v"].mean(), 9.114468, 6)


def test_load_penttila2016():

    result = pyedra.datasets.load_penttila2016()

    assert isinstance(result, pd.DataFrame)
    np.testing.assert_almost_equal(result["alpha"].mean(), 40.951960, 6)
    np.testing.assert_almost_equal(result["phi1"].mean(), 0.491135, 6)
    np.testing.assert_almost_equal(result["phi2"].mean(), 0.610840, 6)
    np.testing.assert_almost_equal(result["phi3"].mean(), 0.213223, 6)
