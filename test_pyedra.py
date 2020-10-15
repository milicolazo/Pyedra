import numpy as np

import pandas as pd

import pyedra

import pytest


@pytest.fixture(scope="session")
def data():
    return pd.read_csv("test_data/testdata_carbognani2019.csv")


@pytest.fixture(scope="session")
def carbognani2019():
    return pd.read_csv("test_data/carbognani2019.csv")


def test_HG_fit(data, carbognani2019):
    noob = data.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.HG_fit(data)

    np.testing.assert_array_equal(noob.nro, result.Asteroid)

    for idx, error in enumerate(result.error_H):
        np.testing.assert_allclose(
            carbognani2019.H[idx], result.H[idx], atol=error
        )
    for idx, error in enumerate(result.error_G):
        np.testing.assert_allclose(
            carbognani2019.G[idx], result.G[idx], atol=error
        )


def test_HG1G2_fit(data, carbognani2019):
    noob = data.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.HG1G2_fit(data)

    np.testing.assert_array_equal(noob.nro, result.Asteroid)

    for idx, error in enumerate(result.error_H12):
        np.testing.assert_allclose(
            carbognani2019.H12[idx], result.H12[idx], atol=error
        )
    for idx, error in enumerate(result.error_G1):
        np.testing.assert_allclose(
            carbognani2019.G1[idx], result.G1[idx], atol=error
        )
    for idx, error in enumerate(result.error_G2):
        np.testing.assert_allclose(
            carbognani2019.G2[idx], result.G2[idx], atol=error
        )


def test_Shev_fit(data, carbognani2019):
    noob = data.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.Shev_fit(data)

    np.testing.assert_array_equal(noob.nro, result.Asteroid)

    for idx, error in enumerate(result.error_V_lin):
        np.testing.assert_allclose(
            carbognani2019.V_lin[idx], result.V_lin[idx], atol=error
        )
    for idx, error in enumerate(result.error_b):
        np.testing.assert_allclose(
            carbognani2019.b[idx], result.b[idx], atol=error
        )
    for idx, error in enumerate(result.error_c):
        np.testing.assert_allclose(
            carbognani2019.c[idx], result.c[idx], atol=error
        )
