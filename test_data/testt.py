import numpy as np
import scipy
import scipy.interpolate
import pandas as pd
import scipy.optimize as optimization

def HG_fit(df):
    """Fit (H,G) system to data from table"""
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)
    size = len(noob)
    nro_column = np.empty(size, dtype=int)
    H_column = np.empty(size)
    error_H_column = np.empty(size)
    G_column = np.empty(size)
    error_G_column = np.empty(size)
    R_column = np.empty(size)

    for idx, nro in enumerate(noob.nro):

        # Filtrar un solo asteroide
        data = df[df["nro"] == nro]

        alfa_list = data["alfa"].to_numpy()
        V_list = data["v"].to_numpy()

        v_fit = 10 ** (-0.4 * V_list)
        alfa_fit = alfa_list * np.pi / 180

        def func(x, a, b):
            return a * np.exp(-3.33 * np.tan(x / 2) ** 0.63) + b * np.exp(
                -1.87 * np.tan(x / 2) ** 1.22
            )

        op, cov = optimization.curve_fit(func, alfa_fit, v_fit)

        a = op[0]
        b = op[1]
        error_a = np.sqrt(np.diag(cov)[0])
        error_b = np.sqrt(np.diag(cov)[1])

        H = -2.5 * np.log10(a + b)
        error_H = 1.0857362 * np.sqrt(error_a ** 2 + error_b ** 2) / (a + b)
        G = b / (a + b)
        error_G = (
            np.sqrt((b * error_a) ** 2 + (a * error_b) ** 2) / (a + b) ** 2
        )

        # Pa decidir el mejor ajuste
        residuals = v_fit - func(alfa_fit, *op)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((v_fit - np.mean(v_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        nro_column[idx] = nro
        H_column[idx] = H
        error_H_column[idx] = error_H
        G_column[idx] = G
        error_G_column[idx] = error_G
        R_column[idx] = r_squared

    # aca fuera del for tenes todas las columnas listas y podes
    # crear tu dataframe resultado
    model_df = pd.DataFrame({
        "Asteroid": nro_column,
        "H": H_column,
        "error_H": error_H_column,
        "G": G_column,
        "error_G": error_G,
        "R": R_column
    })

    return model_df


def Shev_fit(df):
    """Fit Shevchenko system to data from table"""
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)

    for nro in noob.nro:
        2.5 / (np.log(10))
        # Filtrar un solo asteroide
        data = df[df["nro"] == nro]

        alfa_list = data["alfa"].to_numpy()
        V_list = data["v"].to_numpy()

        def func(x, V_lin, b, c):
            return V_lin + c * x - b / (1 + x)

        op = optimization.curve_fit(func, alfa_list, V_list)[0]
        V_lin = op[0]
        b = op[1]
        c = op[2]

        # Pa decidir el mejor ajuste
        residuals = V_list - func(alfa_list, *op)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((V_list - np.mean(V_list)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(V_lin, b, c, r_squared)


def HG1G2_fit(df):
    """Fit (H,G1,G2) system to data from table"""
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)

    for nro in noob.nro:

        # Filtrar un solo asteroide
        data = df[df["nro"] == nro]

        bases = pd.read_csv("Penttila.csv")

        alfa = bases["alfa"].to_numpy()
        phi1 = bases["phi1"].to_numpy()
        phi2 = bases["phi2"].to_numpy()
        phi3 = bases["phi3"].to_numpy()

        y_interp1 = scipy.interpolate.interp1d(alfa, phi1)
        y_interp2 = scipy.interpolate.interp1d(alfa, phi2)
        y_interp3 = scipy.interpolate.interp1d(alfa, phi3)

        fi1 = np.array([])
        fi2 = np.array([])
        fi3 = np.array([])

        for alfa_b in data.alfa:

            p1 = y_interp1(alfa_b)
            fi1 = np.append(fi1, p1)

            p2 = y_interp2(alfa_b)
            fi2 = np.append(fi2, p2)

            p3 = y_interp3(alfa_b)
            fi3 = np.append(fi3, p3)

        def func(X, a, b, c):
            x, y, z = X
            return a * x + b * y + c * z

        v = data["v"].to_numpy()
        v_fit = 10 ** (-0.4 * v)

        op = optimization.curve_fit(func, (fi1, fi2, fi3), v_fit)[0]
        a = op[0]
        b = op[1]
        c = op[2]

        H_1_2 = -2.5 * np.log10(a + b + c)
        G_1 = a / (a + b + c)
        G_2 = b / (a + b + c)

        # Pa decidir el mejor ajuste
        residuals = v_fit - func((fi1, fi2, fi3), *op)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((v_fit - np.mean(v_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(H_1_2, G_1, G_2, r_squared)

df=pd.read_csv('prueba3.csv')
def test_HG():
    result=HG_fit(df)
    for nro in df.nro
        assert result.Asteroid==nro
    
    

"""if __name__ == "__main__":
    print("H-G")
    df=pd.read_csv('prueba3.csv')
    print(HG_fit(df))
    print('----------------------------')
    print("H-G1-G2")
    HG1G2_fit("prueba3.csv")
    print('----------------------------')
    print("Shevchenko model")
    Shev_fit("prueba3.csv")
    print('----------------------------')"""
