#!/usr/bin/env python

# -*- coding: utf-8 -*-


# This file is part of the

#   Pyedra Project (https://github.com/milicolazo/Pyedra/).

# Copyright (c) 2020, Milagros Colazo

# License: MIT

#   Full Text: https://github.com/milicolazo/Pyedra/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""H,G model for Pyedra."""


# =============================================================================
# IMPORTS
# =============================================================================

import attr

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy.optimize as optimization

from . import core

# =============================================================================
# CLASSES
# =============================================================================


@attr.s(frozen=True)
class HGPlot(core.BasePlot):
    """Plots for HG fit."""

    default_plot_kind = "curvefit"

    def curvefit(self, df, ax=None, **kwargs):
        """Plot the phase function using the HG model.

        Parameters
        ----------
        df : ``pandas.DataFrame``
            The dataframe must contain 3 columns as indicated here:
            id (mpc number of the asteroid), alpha (phase angle) and
            v (reduced magnitude in Johnson's V filter).

        ax : ``matplotlib.pyplot.Axis``, (optional)
            Matplotlib axis

        **kwargs :
            Extra variables are not used


        Return
        ------
        ``matplotlib.pyplot.Axis`` :
            The axis where the method draws.

        """

        def fit_y(alpha, H, G):
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

        for idx, m_row in self.pdf.model_df.iterrows():
            data = df[df["id"] == m_row.id]
            v_fit = fit_y(data.alpha, m_row.H, m_row.G)
            ax.plot(data.alpha, v_fit, "--", label=f"Fit {int(m_row.id)}")
            ax.plot(
                data.alpha,
                data.v,
                marker="o",
                linestyle="None",
                label=f"Data {int(m_row.id)}",
            )

        ax.legend(bbox_to_anchor=(1.05, 1))
        return ax


# =============================================================================
# FUNCTIONS
# =============================================================================


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
    lt = core.obs_counter(df, 2)
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

    return core.PyedraFitDataFrame(
        model_df=model_df, plot_cls=HGPlot, model="HG"
    )
