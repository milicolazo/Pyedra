import pandas as pd


def test_HG_fit():
    df=pd.read_csv('prueba3.csv')
    result=HG_fit(df)
    
    for nro in result.nro:
        
        assert result.nro == nro
