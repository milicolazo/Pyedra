#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Pyedra Project (https://github.com/milicolazo/Pyedra/).
# Copyright (c) 2020, Milagros Colazo
# License: MIT
#   Full Text: https://github.com/milicolazo/Pyedra/blob/master/LICENSE

import numpy as np

import pandas as pd

import scipy
import scipy.interpolate
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
        error_G = np.sqrt((b * error_a) ** 2 + (a * error_b) ** 2) / (
            (a + b) ** 2
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

    model_df = pd.DataFrame(
        {
            "Asteroid": nro_column,
            "H": H_column,
            "error_H": error_H_column,
            "G": G_column,
            "error_G": error_G_column,
            "R": R_column,
        }
    )

    return model_df


def Shev_fit(df):
    """Fit Shevchenko system to data from table"""
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)
    size = len(noob)
    nro_column = np.empty(size, dtype=int)
    V_lin_column = np.empty(size)
    error_V_lin_column = np.empty(size)
    b_column = np.empty(size)
    error_b_column = np.empty(size)
    c_column = np.empty(size)
    error_c_column = np.empty(size)
    R_column = np.empty(size)

    for idx, nro in enumerate(noob.nro):

        # Filtrar un solo asteroide
        data = df[df["nro"] == nro]

        alfa_list = data["alfa"].to_numpy()
        V_list = data["v"].to_numpy()

        def func(x, V_lin, b, c):
            return V_lin + c * x - b / (1 + x)

        op, cov = optimization.curve_fit(func, alfa_list, V_list)
        V_lin = op[0]
        b = op[1]
        c = op[2]
        error_V_lin = np.sqrt(np.diag(cov)[0])
        error_b = np.sqrt(np.diag(cov)[1])
        error_c = np.sqrt(np.diag(cov)[2])

        # Pa decidir el mejor ajuste
        residuals = V_list - func(alfa_list, *op)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((V_list - np.mean(V_list)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        nro_column[idx] = nro
        V_lin_column[idx] = V_lin
        error_V_lin_column[idx] = error_V_lin
        b_column[idx] = b
        error_b_column[idx] = error_b
        c_column[idx] = c
        error_c_column[idx] = error_c
        R_column[idx] = r_squared

    model_df = pd.DataFrame(
        {
            "Asteroid": nro_column,
            "V_lin": V_lin_column,
            "error_V_lin": error_V_lin_column,
            "b": b_column,
            "error_b": error_b_column,
            "c": c_column,
            "error_c": error_c_column,
            "R": R_column,
        }
    )

    return model_df


def HG1G2_fit(df):
    """Fit (H,G1,G2) system to data from table"""
    noob = df.drop_duplicates(subset="nro", keep="first", inplace=False)
    size = len(noob)
    nro_column = np.empty(size, dtype=int)
    H_1_2_column = np.empty(size)
    error_H_1_2_column = np.empty(size)
    G_1_column = np.empty(size)
    error_G_1_column = np.empty(size)
    G_2_column = np.empty(size)
    error_G_2_column = np.empty(size)
    R_column = np.empty(size)

    for idx, nro in enumerate(noob.nro):

        # Filtrar un solo asteroide
        data = df[df["nro"] == nro]

        bases = pd.read_csv("data/penttila2016.csv")

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

        op, cov = optimization.curve_fit(func, (fi1, fi2, fi3), v_fit)
        a = op[0]
        b = op[1]
        c = op[2]
        error_a = np.sqrt(np.diag(cov)[0])
        error_b = np.sqrt(np.diag(cov)[1])
        error_c = np.sqrt(np.diag(cov)[2])

        H_1_2 = -2.5 * np.log10(a + b + c)
        error_H_1_2 = (
            1.0857362
            * np.sqrt(error_a ** 2 + error_b ** 2 + error_c ** 2)
            / (a + b + c)
        )
        G_1 = a / (a + b + c)
        error_G_1 = np.sqrt(
            ((b + c) * error_a) ** 2 + (a * error_b) ** 2 + (a * error_c) ** 2
        ) / ((a + b + c) ** 2)
        G_2 = b / (a + b + c)
        error_G_2 = np.sqrt(
            (b * error_a) ** 2 + ((a + c) * error_b) ** 2 + (b * error_c) ** 2
        ) / ((a + b + c) ** 2)

        # Pa decidir el mejor ajuste
        residuals = v_fit - func((fi1, fi2, fi3), *op)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((v_fit - np.mean(v_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        nro_column[idx] = nro
        H_1_2_column[idx] = H_1_2
        error_H_1_2_column[idx] = error_H_1_2
        G_1_column[idx] = G_1
        error_G_1_column[idx] = error_G_1
        G_2_column[idx] = G_2
        error_G_2_column[idx] = error_G_2
        R_column[idx] = r_squared

    model_df = pd.DataFrame(
        {
            "Asteroid": nro_column,
            "H12": H_1_2_column,
            "error_H12": error_H_1_2_column,
            "G1": G_1_column,
            "error_G1": error_G_1_column,
            "G2": G_2_column,
            "error_G2": error_G_2_column,
            "R": R_column,
        }
    )

    return model_df
