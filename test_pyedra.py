import numpy as np

import pandas as pd

import pyedra

import pytest


@pytest.fixture(scope="session")
def ground():
    return pd.read_csv("test_data/testdata_ground.csv")


@pytest.fixture(scope="session")
def carbognani():
    return pd.read_csv("test_data/testdata_carbognani.csv")


def test_HG_fit(ground, carbognani):
    noob = ground.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.HG_fit(ground)

    np.testing.assert_array_equal(noob.nro, result.Asteroid)

    for idx, error in enumerate(result.error_H):
        np.testing.assert_allclose(
            carbognani.H[idx], result.H[idx], atol=error
        )
    for idx, error in enumerate(result.error_G):
        np.testing.assert_allclose(
            carbognani.G[idx], result.G[idx], atol=error
        )


def test_HG1G2_fit(ground, carbognani):
    noob = ground.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.HG1G2_fit(ground)

    np.testing.assert_array_equal(noob.nro, result.Asteroid)

    for idx, error in enumerate(result.error_H12):
        np.testing.assert_allclose(
            carbognani.H12[idx], result.H12[idx], atol=error
        )
    for idx, error in enumerate(result.error_G1):
        np.testing.assert_allclose(
            carbognani.G1[idx], result.G1[idx], atol=error
        )
    for idx, error in enumerate(result.error_G2):
        np.testing.assert_allclose(
            carbognani.G2[idx], result.G2[idx], atol=error
        )


def test_Shev_fit(ground, carbognani):
    noob = ground.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.Shev_fit(ground)

    np.testing.assert_array_equal(noob.nro, result.Asteroid)

    for idx, error in enumerate(result.error_V_lin):
        np.testing.assert_allclose(
            carbognani.V_lin[idx], result.V_lin[idx], atol=error
        )
    for idx, error in enumerate(result.error_b):
        np.testing.assert_allclose(
            carbognani.b[idx], result.b[idx], atol=error
        )
    for idx, error in enumerate(result.error_c):
        np.testing.assert_allclose(
            carbognani.c[idx], result.c[idx], atol=error
        )
