import numpy as np
import scipy
import scipy.interpolate
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons


def creator_gaia(table):
    """Creador de la tabla a partir de Gaia"""
    df=pd.read_csv('prueba2.csv')
    
    #Limpieza del archivo de gaia 
    #Borrar las filas vacías
    df=df.dropna(how='any')
    #Borrar las magnitudes repetidas
    df=df.drop_duplicates(subset='g_mag', keep='first', inplace=False)
    #Ordenar por número mpc. Para cada nro mpc está ordenado por fecha
    dff=df.sort_values(by=['number_mp', 'epoch_utc'])
    #Agrego la columna de epoca utc pasada a JD
    jd=dff['epoch_utc'] + 2455197.5
    dff.insert(2,'jd',jd)


    #Borro valores repetidos para tener una lista de nro_mpc
    noob = df.drop_duplicates(subset="number_mp", keep="first", inplace=False)
    
    appended_data = []

    #Arranca el loop que me dará para cada asteroide un archivo con: (nro-epoch_utc-JD-g_inst-r-delta-alfa)
    for nro in noob.number_mp:
        
        #Filtrar un solo asteroide
        data = dff[dff["number_mp"] == nro]
               
        #EFEMERIDES-----------------------------------------------------------------------

        #Descarga de las efemérides
        obj = Horizons(id=nro, location='@Gaia',
                        epochs={'start':'2014-07-10',
                                'stop':'2016-06-20',
                                'step':'1d'})
                        
        eph = obj.ephemerides()
        
        #Los datos que necesito de las efemérides
        jdd=eph.columns['datetime_jd']
        r=eph.columns['r']
        delta=eph.columns['delta']
        alpha=eph.columns['alpha']

        efem = pd.DataFrame({'jd': jdd,'r': r, 'delta': delta, 'alpha':alpha})

        #Buscar las filas coincidentes entre archivos segun JD
        jd_list = data["jd"].tolist()
        bb = np.array([])
        for i in range (len(jd_list)):
            b=int(jd_list[i])+.5
            bb = np.append(bb, b)

        r1 = np.array([])
        delta1 = np.array([])
        alfa1= np.array ([])

        for i in range (len(bb)):
            lista=efem[efem.jd == bb[i]]

            rr = lista['r'].tolist()
            r1= np.append(r1,rr)

            dd= lista['delta'].tolist()
            delta1= np.append(delta1,dd)

            aa=lista['alpha'].tolist()
            alfa1=np.append(alfa1,aa)

        #Agrego los datos a la tabla
        data.insert(4,'r',r1)
        data.insert(5,'delta',delta1)
        data.insert(6,'alfa',alfa1)

        #Necesito quedarme con la magnitud  mas chica de cada dia
        data=data.sort_values(by=['epoch_utc','g_mag'],ascending=True)
        data=data.drop_duplicates(subset='r', keep='first', inplace=False)
        
        data['g_red']= data['g_mag'] - 5 * np.log10(data['r']*data['delta'])
                
                
        styp = obj.ephemerides(get_raw_response=True)
        with open('ef.txt', 'w') as f:
            f.write(styp)
        
        with open('ef.txt') as file:    
            file = file.read()
            x=file.find('STYP')
            tax=file[x+6]        
        
        if tax == 'n':
            tax='No' #Asigna un V-R promedio

        
        spec = pd.DataFrame({'Spectral Type': ['No','A','B','C','M','D','F','G','Q','R','S','L', 'K','P','T','V','X', 'E', 'O'],
                             'V-R':[0.42875, 0.560, 0.361 ,0.376 ,0.376, 0.464 ,0.366 ,0.370 ,0.424,0.479, 0.475,0.475, 0.475, 0.475, 0.447, 0.413 ,0.410,0.410,0.410]})


        VR = spec[spec['Spectral Type']== tax]
        VR=VR.iat[0,1]

        #Cálculo de V
        data['V']= data['g_red'] + 0.008 + 0.190 * (VR) + 0.575 * (VR)**2 
        
        dat=data[['number_mp','alfa','V']]
        dat=dat.sort_values(by=['alfa'])
        dat=dat.rename(columns = {'number_mp': 'nro', 'V': 'v'})
        
        appended_data.append(dat)
    
    appended_data = pd.concat(appended_data)
    appended_data.to_csv('prueba3.csv',index=False)
    
if __name__ == "__main__":
    creator_gaia("prueba2.csv")
