import numpy as np
import scipy
import scipy.interpolate
import pandas as pd
import matplotlib.pyplot as plt


def HG_fit(table, graph):
    """Fit (H,G) system to data from table"""
    df = pd.read_csv(table)
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)

    for nro in noob.nro:

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

        import scipy.optimize as optimization

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

        print(
            "Asteroid",
            nro,
            "H=",
            H,
            "Error H=",
            error_H,
            "G=",
            G,
            "Error G=",
            error_G,
            "R",
            r_squared,
        )

        if graph is True:

            def MODEL(x, H, G):
                y = H - 2.5 * np.log10(
                    (1 - G) * np.exp(-3.33 * np.tan(x / 2) ** 0.63)
                    + G * np.exp(-1.87 * np.tan(x / 2) ** 1.22)
                )
                return y

            modely = MODEL(alfa_fit, H, G)
            plt.figure()
            plt.title("Asteroid " + str(nro))
            plt.plot(alfa_list, modely, "--", color="black", label="H,G")
            plt.plot(
                alfa_list,
                V_list,
                linestyle="none",
                marker="o",
                markersize=5,
                label="Gaia observations",
                color="blue",
            )
            plt.xlabel("phase angle $\phi$ (deg) ", fontsize=15)
            plt.xticks(fontsize=12)
            plt.ylabel("V($\phi$)", fontsize=12)
            plt.yticks(fontsize=12)
            plt.gca().invert_yaxis()
            plt.legend(fontsize=12)
            plt.legend()
            plt.show()
            # plt.savefig('asteroid'+str(nro)+'.png')


def Shev_fit(table):
    """Fit Shevchenko system to data from table"""
    df = pd.read_csv(table)
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)

    for nro in noob.nro:
        2.5 / (np.log(10))
        # Filtrar un solo asteroide
        data = df[df["nro"] == nro]

        alfa_list = data["alfa"].to_numpy()
        V_list = data["v"].to_numpy()

        def func(x, V_lin, b, c):
            return V_lin + c * x - b / (1 + x)

        import scipy.optimize as optimization

        op = optimization.curve_fit(func, alfa_list, V_list)[0]
        V_lin = op[0]
        b = op[1]
        c = op[2]

        # Pa decidir el mejor ajuste
        residuals = V_list - func(alfa_list, *op)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((V_list - np.mean(V_list)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        # print(r_squared)

        print(
            "Asteroid", nro, "V_lin=", V_lin, "b=", b, "c=", c, "R", r_squared
        )


def HG1G2_fit(table):

    """Fit (H,G1,G2) system to data from table"""
    df = pd.read_csv(table)
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

        import scipy.optimize as optimization

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
        # print(r_squared)

        print(
            "Asteroid",
            nro,
            "H=",
            H_1_2,
            "G1=",
            G_1,
            "G2=",
            G_2,
            "R",
            r_squared,
        )


#if __name__ == "__main__":
# print("H-G")
#HG_fit("prueba3.csv", True)    
# print('----------------------------')
# print("H-G1-G2")
# HG1G2_fit("prueba3.csv")
# print('----------------------------')
# print("Shevchenko model")
# Shev_fit("prueba3.csv")
# print('----------------------------')
