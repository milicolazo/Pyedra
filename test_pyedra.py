import pyedra
import pandas as pd
import pytest
import numpy as np

@pytest.fixture(scope="session")
def dataset_3():
    return pd.read_csv('test_data/testdata_ground.csv')
    
c = pd.read_csv('test_data/testdata_carbognani.csv')

def test_HG_fit(dataset_3):
    df = dataset_3
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.HG_fit(df)
    np.testing.assert_array_equal(noob.nro, result.Asteroid)
    for idx, e in enumerate(c.error_H):
        if result.R[idx] < 0.5:
            continue
        np.testing.assert_allclose(c.H[idx],result.H[idx],rtol=e)
        np.testing.assert_allclose(c.G[idx],result.G[idx],rtol=e)

def test_HG1G2_fit(dataset_3):
    df = dataset_3
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.HG1G2_fit(df)
    np.testing.assert_array_equal(noob.nro, result.Asteroid)
    for idx, e in enumerate(result.error_H12):
        if result.R[idx] < 0.5:
            continue
        np.testing.assert_allclose(c.H12[idx],result.H12[idx],rtol=e)
        np.testing.assert_allclose(c.G1[idx],result.G1[idx],rtol=e)
        np.testing.assert_allclose(c.G2[idx],result.G2[idx],rtol=e)
