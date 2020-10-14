import numpy as np

import pandas as pd

import pyedra

import pytest


@pytest.fixture(scope="session")
def dataset_3():
    return pd.read_csv("test_data/testdata_ground.csv")


c = pd.read_csv("test_data/testdata_carbognani.csv")


def test_HG_fit(dataset_3):
    df = dataset_3
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.HG_fit(df)
    np.testing.assert_array_equal(noob.nro, result.Asteroid)
    for idx, e in enumerate(result.error_H):
        np.testing.assert_allclose(c.H[idx], result.H[idx], rtol=e)
    for idx, e in enumerate(result.error_G):
        np.testing.assert_allclose(c.G[idx], result.G[idx], rtol=e)


def test_HG1G2_fit(dataset_3):
    df = dataset_3
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.HG1G2_fit(df)
    np.testing.assert_array_equal(noob.nro, result.Asteroid)
    for idx, e in enumerate(result.error_H12):
        np.testing.assert_allclose(c.H12[idx], result.H12[idx], rtol=e)
    for idx, e in enumerate(result.error_G1):
        np.testing.assert_allclose(c.G1[idx], result.G1[idx], rtol=e)
    for idx, e in enumerate(result.error_G2):
        np.testing.assert_allclose(c.G2[idx], result.G2[idx], rtol=e)


def test_Shev_fit(dataset_3):
    df = dataset_3
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.Shev_fit(df)
    np.testing.assert_array_equal(noob.nro, result.Asteroid)
    for idx, e in enumerate(result.error_V_lin):
        np.testing.assert_allclose(c.V_lin[idx], result.V_lin[idx], rtol=e)
    for idx, e in enumerate(result.error_b):
        np.testing.assert_allclose(c.b[idx], result.b[idx], rtol=e)
    for idx, e in enumerate(result.error_c):
        np.testing.assert_allclose(c.c[idx], result.c[idx], rtol=e)
