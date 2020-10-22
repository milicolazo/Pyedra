#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Pyedra Project (https://github.com/milicolazo/Pyedra/).
# Copyright (c) 2020, Milagros Colazo
# License: MIT
#   Full Text: https://github.com/milicolazo/Pyedra/blob/master/LICENSE

import core

import numpy as np

import pandas as pd

import pytest


@pytest.fixture(scope="session")
def input_data():

    return pd.read_csv(core.CARBOGNANI2019_PATH)


def test_HG_fit(input_data):

    result = core.HG_fit(input_data)

    expected = pd.DataFrame(
        {
            "id": {0: 85, 1: 208, 2: 236, 3: 306, 4: 313, 5: 338, 6: 522},
            "H": {0: 7.52, 1: 9.2, 2: 8.0, 3: 8.79, 4: 8.87, 5: 8.51, 6: 9.0},
            "G": {
                0: 0.0753,
                1: 0.2761,
                2: 0.0715,
                3: 0.2891,
                4: 0.1954,
                5: -0.0812,
                6: 0.1411,
            },
        }
    )

    for idx, e_row in expected.iterrows():
        r_row = result[result.id == e_row.id].iloc[0]
        np.testing.assert_array_equal(r_row.id, e_row.id)
        np.testing.assert_allclose(r_row.H, e_row.H, atol=r_row.error_H)
        np.testing.assert_allclose(r_row.G, e_row.G, atol=r_row.error_G)


def test_HG1G2_fit(input_data):

    result = core.HG1G2_fit(input_data)

    expected = pd.DataFrame(
        {
            "id": {0: 85, 1: 208, 2: 236, 3: 306, 4: 313, 5: 338, 6: 522},
            "H12": {
                0: 7.41,
                1: 8.92,
                2: 7.82,
                3: 8.04,
                4: 8.88,
                5: 8.41,
                6: 9.07,
            },
            "G1": {
                0: 0.3358,
                1: -0.3116,
                2: 0.1155,
                3: -0.1309,
                4: 0.624,
                5: 0.5607,
                6: 0.7302,
            },
            "G2": {
                0: 0.2147,
                1: 0.6598,
                2: 0.3436,
                3: 0.3624,
                4: 0.151,
                5: -0.0037,
                6: 0.0879,
            },
        }
    )

    for idx, e_row in expected.iterrows():
        r_row = result[result.id == e_row.id].iloc[0]
        np.testing.assert_array_equal(r_row.id, e_row.id)
        np.testing.assert_allclose(r_row.H12, e_row.H12, atol=r_row.error_H12)
        np.testing.assert_allclose(r_row.G1, e_row.G1, atol=r_row.error_G1)
        np.testing.assert_allclose(r_row.G2, e_row.G2, atol=r_row.error_G2)


def test_Shev_fit(input_data):

    result = core.Shev_fit(input_data)

    expected = pd.DataFrame(
        {
            "id": {0: 85, 1: 208, 2: 236, 3: 306, 4: 313, 5: 338, 6: 522},
            "V_lin": {
                0: 7.96,
                1: 9.74,
                2: 8.56,
                3: 9.6,
                4: 9.13,
                5: 9.0,
                6: 9.29,
            },
            "b": {
                0: 0.7,
                1: 0.95,
                2: 0.97,
                3: 3.23,
                4: 0.3,
                5: 0.9,
                6: 0.39,
            },
            "c": {
                0: 0.035,
                1: 0.014,
                2: 0.026,
                3: 0.009,
                4: 0.037,
                5: 0.045,
                6: 0.04,
            },
        }
    )

    for idx, e_row in expected.iterrows():
        r_row = result[result.id == e_row.id].iloc[0]
        np.testing.assert_array_equal(r_row.id, e_row.id)
        np.testing.assert_allclose(
            r_row.V_lin, e_row.V_lin, atol=r_row.error_V_lin
        )
        np.testing.assert_allclose(r_row.b, e_row.b, atol=r_row.error_b)
        np.testing.assert_allclose(r_row.c, e_row.c, atol=r_row.error_c)
