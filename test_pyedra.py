import numpy as np

import pandas as pd

import pyedra

import pytest


@pytest.fixture(scope="session")
def input_data_data():
    """Carbognani, A., Cellino, A., & Caminiti, S. (2019). New phase-magnitude
    curves for some main belt asteroids, fit of different photometric systems
    and calibration of the albedo-Photometry relation. Planetary and Space
    Science, 169, 15-34."""
    return pd.read_csv("data/inputdata_carbognani2019.csv")


@pytest.fixture(scope="session")
def results():
    """Carbognani, A., Cellino, A., & Caminiti, S. (2019). New phase-magnitude
    curves for some main belt asteroids, fit of different photometric systems
    and calibration of the albedo-Photometry relation. Planetary and Space
    Science, 169, 15-34."""
    return pd.read_csv("data/results_carbognani2019.csv")


def test_HG_fit(input_data, results):
    noob = input_data.drop_duplicates(
        subset="nro", keep="first", inplace=False
    )
    result = pyedra.HG_fit(input_data)

    np.testing.assert_array_equal(noob.nro, result.Asteroid)

    for idx, error in enumerate(result.error_H):
        np.testing.assert_allclose(results.H[idx], result.H[idx], atol=error)
    for idx, error in enumerate(result.error_G):
        np.testing.assert_allclose(results.G[idx], result.G[idx], atol=error)


def test_HG1G2_fit(input_data, results):
    noob = input_data.drop_duplicates(
        subset="nro", keep="first", inplace=False
    )
    result = pyedra.HG1G2_fit(input_data)

    np.testing.assert_array_equal(noob.nro, result.Asteroid)

    for idx, error in enumerate(result.error_H12):
        np.testing.assert_allclose(
            results.H12[idx], result.H12[idx], atol=error
        )
    for idx, error in enumerate(result.error_G1):
        np.testing.assert_allclose(results.G1[idx], result.G1[idx], atol=error)
    for idx, error in enumerate(result.error_G2):
        np.testing.assert_allclose(results.G2[idx], result.G2[idx], atol=error)


def test_Shev_fit(input_data, results):
    noob = input_data.drop_duplicates(
        subset="nro", keep="first", inplace=False
    )
    result = pyedra.Shev_fit(input_data)

    np.testing.assert_array_equal(noob.nro, result.Asteroid)

    for idx, error in enumerate(result.error_V_lin):
        np.testing.assert_allclose(
            results.V_lin[idx], result.V_lin[idx], atol=error
        )
    for idx, error in enumerate(result.error_b):
        np.testing.assert_allclose(results.b[idx], result.b[idx], atol=error)
    for idx, error in enumerate(result.error_c):
        np.testing.assert_allclose(results.c[idx], result.c[idx], atol=error)
