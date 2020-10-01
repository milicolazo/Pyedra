import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
import pandas as pd
from astroquery.jplhorizons import Horizons
import csv
import os
from math import nan

class Core:
    def __init__(self):
        """Acá irían algunas variables o things"""
        
        
    def HG_fit(self,table):
        """Fit (H,G) system to data from table"""
        d=pd.read_csv(table)

        #Borro valores repetidos para tener una lista de nro_mpc
        noob=d.drop_duplicates(subset='nro', keep='first', inplace=False)
        nro = noob["nro"].tolist()

        for j in range (len(nro)):
            
            #Filtrar un solo asteroide
            data = d[d['nro']==nro[j]]
        
            alfa_list = data["alfa"].tolist()
            alfa_list = np.array(alfa_list)
            V_list = data["v"].tolist()
            V_list = np.array(V_list)
            
            v_fit=10**(-0.4*V_list)
            alfa_fit=alfa_list*np.pi/180
            
            def func(x, a, b):
                return (a*np.exp(-3.33*np.tan(x/2)**0.63) + b*np.exp(-1.87*np.tan(x/2)**1.22))

            import scipy.optimize as optimization
            op=optimization.curve_fit(func, alfa_fit, v_fit)[0]
            a=op[0]
            b=op[1]
            
            H=-2.5*np.log10(a+b)
            G=b/(a+b)
            
            print('Asteroid',nro[j],'H=',H,'G=',G)
    
    def Shev_fit(self,table):
        """Fit Shevchenko system to data from table"""
        d=pd.read_csv(table)
        noob=d.drop_duplicates(subset='nro', keep='first', inplace=False)
        nro = noob["nro"].tolist()

        for j in range (len(nro)):
            
            #Filtrar un solo asteroide
            data = d[d['nro']==nro[j]]
        
            alfa_fit = data["alfa"].tolist()
            alfa_fit = np.array(alfa_fit)
            v_fit = data["v"].tolist()
            v_fit = np.array(v_fit)
            
            def func(x, V_lin, b, c):
                return (V_lin+c*x-b/(1+x))

            import scipy.optimize as optimization
            op=optimization.curve_fit(func, alfa_fit, v_fit)[0]
            V_lin=op[0]
            b=op[1]
            c=op[2]

            print('Asteroid',nro[j],'V_lin=',V_lin,'b=',b, 'c=',c)
            
    def HG1G2_fit(self,table):
            
        """Fit (H,G1,G2) system to data from table"""
        d=pd.read_csv(table)

        #Borro valores repetidos para tener una lista de nro_mpc
        noob=d.drop_duplicates(subset='nro', keep='first', inplace=False)
        nro = noob["nro"].tolist()

        for j in range(len(nro)):
            
            #Filtrar un solo asteroide
            data = d[d['nro']==nro[j]]
            
            bases=pd.read_csv('Penttila.csv')
            
            alfa = np.asarray(bases["alfa"].tolist())
            phi1=np.asarray(bases["phi1"].tolist())
            phi2=np.asarray(bases["phi2"].tolist())
            phi3=np.asarray(bases["phi3"].tolist())

            y_interp1 = scipy.interpolate.interp1d(alfa, phi1)
            y_interp2 = scipy.interpolate.interp1d(alfa, phi2)
            y_interp3 = scipy.interpolate.interp1d(alfa, phi3)

            alfa_b = data["alfa"].tolist()

            fi1=np.array([])
            fi2=np.array([])
            fi3=np.array([])
            

            for k in range (len(alfa_b)):
                
                p1=y_interp1(alfa_b[k])
                fi1=np.append(fi1,p1)

                p2=y_interp2(alfa_b[k])
                fi2=np.append(fi2,p2)

                p3=y_interp3(alfa_b[k])
                fi3=np.append(fi3,p3)
                
            def func(X, a, b, c):
                x,y,z = X
                return a*x + b*y + c*z
            
            v=np.asarray(data["v"].tolist())
            v_fit=10**(-0.4*v)
            
            import scipy.optimize as optimization
            op=optimization.curve_fit(func, (fi1,fi2,fi3), v_fit)[0]
            a=op[0]
            b=op[1]
            c=op[2]
        
            H_1_2=-2.5*np.log10(a+b+c)

            G_1=a/(a+b+c)
            
            G_2=b/(a+b+c)
        
            print('Asteroid',nro[j],'H=',H_1_2,'G1=',G_1, 'G2=',G_2)

if __name__ == "__main__":
    krixi=Core()
    print('H-G')
    tabla=krixi.HG_fit('prueba1.csv')
    print('H-G1-G2')
    tabla=krixi.HG1G2_fit('prueba1.csv')
    print('Shevchenko model')
    tabla=krixi.Shev_fit('prueba1.csv')
    
        
