#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Pyedra Project (https://github.com/milicolazo/Pyedra/).
# Copyright (c) 2020, Milagros Colazo
# License: MIT
#   Full Text: https://github.com/milicolazo/Pyedra/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Implementation of phase function for asteroids."""

# =============================================================================
# IMPORTS
# =============================================================================

import attr

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy
import scipy.interpolate
import scipy.optimize as optimization

from . import datasets

# ============================================================================
# CLASSES
# ============================================================================


@attr.s(frozen=True)
class PyedraFitDataFrame:
    """Initialize a dataframe model_df to which we can apply function a."""

    model_df = attr.ib()

    def __getattr__(self, a):
        """Return the dataframe after applying the "a" function."""
        return getattr(self.model_df, a)

    def plot(self, df, ax=None, **kwargs):
        """Return a graph."""
        noob = df.drop_duplicates(subset="id", keep="first", inplace=False)

        def fit(alpha, H, G):
            x = alpha * np.pi / 180
            y = H - 2.5 * np.log10(
                (1 - G) * np.exp(-3.33 * np.tan(x / 2) ** 0.63)
                + G * np.exp(-1.87 * np.tan(x / 2) ** 1.22)
            )
            return y

        if ax is None:
            ax = plt.gca()

        ax.invert_yaxis()
        ax.set_title("Phase curves")
        ax.set_xlabel("Phase angle")
        ax.set_ylabel("V")

        for idx, n_row in noob.iterrows():
            data = df[df["id"] == n_row.id]
            m_row = self.model_df[self.model_df.id == n_row.id].iloc[0]
            v_fit = fit(data.alpha, m_row.H, m_row.G)
            ax.plot(data.alpha, v_fit, "--")
            ax.plot(data.alpha, data.v, marker="o", linestyle="None")
        return ax


# ============================================================================
# FUNCTIONS
# ============================================================================


def obs_counter(df, obs):
    """Count the observations. A minimum is needed to the fits.

    Parameters
    ----------
    df: ``pandas.DataFrame``
        The dataframe must contain 3 columns as indicated here:
        id (mpc number of the asteroid), alpha (phase angle) and
        v (reduced magnitude in Johnson's V filter).

    obs: int
        Minimum number of observations needed to perform the fit.

    Return
    ------
    out: ndarray
        Numpy array containing the asteroids whose number of
        observations is less than obs.
    """
    df_cnt = df.groupby("id").count()
    lt_idx = df_cnt[df_cnt.alpha < obs].index
    return lt_idx.to_numpy()


# ============================================================================
# H, G
# ============================================================================


def _HGmodel(x, a, b):
    return a * np.exp(-3.33 * np.tan(x / 2) ** 0.63) + b * np.exp(
        -1.87 * np.tan(x / 2) ** 1.22
    )


def HG_fit(df):
    """Fit (H-G) system to data from table.

    HG_fit calculates the H and G parameters of the phase function
    following the procedure described in [1]_ .

    Parameters
    ----------
    df: ``pandas.DataFrame``
        The dataframe must contain 3 columns as indicated here:
        id (mpc number of the asteroid), alpha (phase angle) and
        v (reduced magnitude in Johnson's V filter).

    Returns
    -------
    model_df: ``pandas.DataFrame``
        The output dataframe contains six columns: id (mpc number of
        the asteroid), H (absolute magnitude returned by the fit),
        H error (fit H parameter error), G (slope parameter returned by
        the fit), G error (fit G parameter error) and R (fit
        determination coefficient).

    References
    ----------
    .. [1] Muinonen K., Belskaya I. N., Cellino A., Delbò M.,
       Levasseur-Regourd A.-C.,Penttilä A., Tedesco E. F., 2010,
       Icarus, 209, 542.
    """
    lt = obs_counter(df, 2)
    if len(lt):
        lt_str = " - ".join(str(idx) for idx in lt)
        raise ValueError(
            f"Some asteroids has less than 2 observations: {lt_str}"
        )

    noob = df.drop_duplicates(subset="id", keep="first", inplace=False)
    size = len(noob)
    id_column = np.empty(size, dtype=int)
    H_column = np.empty(size)
    error_H_column = np.empty(size)
    G_column = np.empty(size)
    error_G_column = np.empty(size)
    R_column = np.empty(size)

    for idx, id in enumerate(noob.id):

        data = df[df["id"] == id]

        alpha_list = data["alpha"].to_numpy()
        V_list = data["v"].to_numpy()

        v_fit = 10 ** (-0.4 * V_list)
        alpha_fit = alpha_list * np.pi / 180

        op, cov = optimization.curve_fit(_HGmodel, alpha_fit, v_fit)

        a, b = op
        error_a, error_b = np.sqrt(np.diag(cov))

        H = -2.5 * np.log10(a + b)
        error_H = 1.0857362 * np.sqrt(error_a ** 2 + error_b ** 2) / (a + b)
        G = b / (a + b)
        error_G = np.sqrt((b * error_a) ** 2 + (a * error_b) ** 2) / (
            (a + b) ** 2
        )

        residuals = v_fit - _HGmodel(alpha_fit, *op)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((v_fit - np.mean(v_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        id_column[idx] = id
        H_column[idx] = H
        error_H_column[idx] = error_H
        G_column[idx] = G
        error_G_column[idx] = error_G
        R_column[idx] = r_squared

    model_df = pd.DataFrame(
        {
            "id": id_column,
            "H": H_column,
            "error_H": error_H_column,
            "G": G_column,
            "error_G": error_G_column,
            "R": R_column,
        }
    )

    return PyedraFitDataFrame(model_df=model_df)


# ============================================================================
# Shevchenko
# ============================================================================


def _Shev_model(x, V_lin, b, c):
    return V_lin + c * x - b / (1 + x)


def Shev_fit(df):
    """Fit Shevchenko equation to data from table.

    Shev_fit calculates parameters of the three-parameter empirical
    function proposed by Schevchenko [2]_, [3]_ .

    Parameters
    ----------
    df: ``pandas.DataFrame``
        The dataframe must contain 3 columns as indicated here:
        id (mpc number of the asteroid), alpha (phase angle) and
        v (reduced magnitude in Johnson's V filter).

    Returns
    -------
    model_df: ``pandas.DataFrame``
        The output dataframe contains six columns: id (mpc number of
        the asteroid), V_lin (magnitude calculated by linear
        extrapolation to zero), V_lin error (fit V_lin parameter
        error), b (fit parameter characterizing the opposition efect
        amplitude), b error (fit b parameter error), c (fit parameter
        describing the linear part of the magnitude phase dependence),
        c error (fit c parameter error) [4]_ and R (fit determination
        coefficient).

    References
    ----------
    .. [2] Shevchenko, V. G. 1996. Analysis of the asteroid phase
       dependences of brightness. Lunar Planet Sci. XXVII, 1086.

    .. [3] Shevchenko, V. G. 1997. Analysis of asteroid brightness
       phase relations. Solar System Res. 31, 219-224.

    .. [4] Belskaya, I. N., Shevchenko, V. G., 2000. Opposition effect
       of asteroids. Icarus 147, 94-105.
    """
    lt = obs_counter(df, 3)
    if len(lt):
        lt_str = " - ".join(str(idx) for idx in lt)
        raise ValueError(
            f"Some asteroids has less than 3 observations: {lt_str}"
        )

    noob = df.drop_duplicates(subset="id", keep="first", inplace=False)
    size = len(noob)
    id_column = np.empty(size, dtype=int)
    V_lin_column = np.empty(size)
    error_V_lin_column = np.empty(size)
    b_column = np.empty(size)
    error_b_column = np.empty(size)
    c_column = np.empty(size)
    error_c_column = np.empty(size)
    R_column = np.empty(size)

    for idx, id in enumerate(noob.id):

        data = df[df["id"] == id]

        alpha_list = data["alpha"].to_numpy()
        V_list = data["v"].to_numpy()

        op, cov = optimization.curve_fit(_Shev_model, alpha_list, V_list)

        V_lin, b, c = op
        error_V_lin, error_b, error_c = np.sqrt(np.diag(cov))

        residuals = V_list - _Shev_model(alpha_list, *op)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((V_list - np.mean(V_list)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        id_column[idx] = id
        V_lin_column[idx] = V_lin
        error_V_lin_column[idx] = error_V_lin
        b_column[idx] = b
        error_b_column[idx] = error_b
        c_column[idx] = c
        error_c_column[idx] = error_c
        R_column[idx] = r_squared

    model_df = pd.DataFrame(
        {
            "id": id_column,
            "V_lin": V_lin_column,
            "error_V_lin": error_V_lin_column,
            "b": b_column,
            "error_b": error_b_column,
            "c": c_column,
            "error_c": error_c_column,
            "R": R_column,
        }
    )

    return PyedraFitDataFrame(model_df=model_df)


# ============================================================================
# H, G1, G2
# ============================================================================


def _HG1G2_model(X, a, b, c):
    x, y, z = X
    return a * x + b * y + c * z


def HG1G2_fit(df):
    """Fit (H-G1-G2) system to data from table.

    HG1G2_fit calculates the H,G1 and G2 parameters of the phase
    function following the procedure described in [5]_ .

    Parameters
    ----------
    df: ``pandas.DataFrame``
        The dataframe must contain 3 columns as indicated here:
        id (mpc number of the asteroid), alpha (phase angle) and
        v (reduced magnitude in Johnson's V filter).

    Returns
    -------
    model_df: ``pandas.DataFrame``
        The output dataframe contains eight columns: id (mpc number of
        the asteroid), H (absolute magnitude returned by the fit),
        H error (fit H parameter error), G1 (G1 parameter returned by
        the fit), G1 error (fit G1 parameter error), G2 (G2 parameter
        returned bythe fit), G2 error (fit G2 parameter error), and R
        (fit determination coefficient).

    References
    ----------
    .. [5] Muinonen K., Belskaya I. N., Cellino A., Delbò M.,
       Levasseur-Regourd A.-C.,Penttilä A., Tedesco E. F., 2010,
       Icarus, 209, 542.
    """
    lt = obs_counter(df, 3)
    if len(lt):
        lt_str = " - ".join(str(idx) for idx in lt)
        raise ValueError(
            f"Some asteroids has less than 3 observations: {lt_str}"
        )

    noob = df.drop_duplicates(subset="id", keep="first", inplace=False)
    size = len(noob)
    id_column = np.empty(size, dtype=int)
    H_1_2_column = np.empty(size)
    error_H_1_2_column = np.empty(size)
    G_1_column = np.empty(size)
    error_G_1_column = np.empty(size)
    G_2_column = np.empty(size)
    error_G_2_column = np.empty(size)
    R_column = np.empty(size)

    penttila2016 = datasets.load_penttila2016()

    alpha = penttila2016["alpha"].to_numpy()
    phi1 = penttila2016["phi1"].to_numpy()
    phi2 = penttila2016["phi2"].to_numpy()
    phi3 = penttila2016["phi3"].to_numpy()

    y_interp1 = scipy.interpolate.interp1d(alpha, phi1)
    y_interp2 = scipy.interpolate.interp1d(alpha, phi2)
    y_interp3 = scipy.interpolate.interp1d(alpha, phi3)

    for idx, id in enumerate(noob.id):

        data = df[df["id"] == id]

        fi1 = np.array([])
        fi2 = np.array([])
        fi3 = np.array([])

        for alpha_b in data.alpha:

            p1 = y_interp1(alpha_b)
            fi1 = np.append(fi1, p1)

            p2 = y_interp2(alpha_b)
            fi2 = np.append(fi2, p2)

            p3 = y_interp3(alpha_b)
            fi3 = np.append(fi3, p3)

        v = data["v"].to_numpy()
        v_fit = 10 ** (-0.4 * v)

        op, cov = optimization.curve_fit(_HG1G2_model, (fi1, fi2, fi3), v_fit)

        a, b, c = op
        error_a, error_b, error_c = np.sqrt(np.diag(cov))

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

        residuals = v_fit - _HG1G2_model((fi1, fi2, fi3), *op)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((v_fit - np.mean(v_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        id_column[idx] = id
        H_1_2_column[idx] = H_1_2
        error_H_1_2_column[idx] = error_H_1_2
        G_1_column[idx] = G_1
        error_G_1_column[idx] = error_G_1
        G_2_column[idx] = G_2
        error_G_2_column[idx] = error_G_2
        R_column[idx] = r_squared

    model_df = pd.DataFrame(
        {
            "id": id_column,
            "H12": H_1_2_column,
            "error_H12": error_H_1_2_column,
            "G1": G_1_column,
            "error_G1": error_G_1_column,
            "G2": G_2_column,
            "error_G2": error_G_2_column,
            "R": R_column,
        }
    )

    return PyedraFitDataFrame(model_df=model_df)
