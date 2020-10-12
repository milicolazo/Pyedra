import pyedra
import pandas as pd
import pytest
import numpy as np

@pytest.fixture(scope="session")
def dataset_3():
    return pd.read_csv('test_data/prueba3.csv')


def test_HG_fit(dataset_3):
    df = dataset_3
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)
    result = pyedra.HG_fit(df)
    np.testing.assert_array_equal(noob.nro, result.Asteroid)
    
  
