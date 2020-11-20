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

import pyedra.datasets

# =============================================================================
# TESTS
# =============================================================================


def test_obs_counter(carbognani2019):

    result = pyedra.obs_counter(carbognani2019, 8)

    expected = [85, 208, 306, 313, 338, 522]

    np.testing.assert_array_almost_equal(result, expected, 6)
