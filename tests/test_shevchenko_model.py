#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Pyedra Project (https://github.com/milicolazo/Pyedra/).
# Copyright (c) 2020, Milagros Colazo
# License: MIT
#   Full Text: https://github.com/milicolazo/Pyedra/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

from unittest import mock

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pandas as pd

import pyedra.datasets

import pytest


# ------------------------------------------------------------------------------
# Shevchenko
# ------------------------------------------------------------------------------


def test_raises_Shev(bad_data):
    with pytest.raises(ValueError):
        pyedra.Shev_fit(bad_data)


def test_PyedraFitDataFrame_Shev(carbognani2019):
    pdf = pyedra.Shev_fit(carbognani2019)

    np.testing.assert_array_equal(pdf.id, pdf.model_df.id)
    np.testing.assert_array_equal(pdf.V_lin, pdf.model_df.V_lin)
    np.testing.assert_array_equal(pdf.b, pdf.model_df.b)
    np.testing.assert_array_equal(pdf.c, pdf.model_df.c)


def test_Shev_fit(carbognani2019):

    result = pyedra.Shev_fit(carbognani2019).model_df

    expected = pd.DataFrame(
        {
            "id": {0: 85, 1: 208, 2: 236, 3: 306, 4: 313, 5: 338, 6: 522},
            "V_lin": {
                0: 7.96,
                1: 9.74,
                2: 8.56,
                3: 9.6,
                4: 9.13,
                5: 9.0,
                6: 9.29,
            },
            "b": {
                0: 0.7,
                1: 0.95,
                2: 0.97,
                3: 3.23,
                4: 0.3,
                5: 0.9,
                6: 0.39,
            },
            "c": {
                0: 0.035,
                1: 0.014,
                2: 0.026,
                3: 0.009,
                4: 0.037,
                5: 0.045,
                6: 0.04,
            },
        }
    )

    for idx, e_row in expected.iterrows():
        r_row = result[result.id == e_row.id].iloc[0]
        np.testing.assert_array_equal(r_row.id, e_row.id)
        np.testing.assert_allclose(
            r_row.V_lin, e_row.V_lin, atol=r_row.error_V_lin
        )
        np.testing.assert_allclose(r_row.b, e_row.b, atol=r_row.error_b)
        np.testing.assert_allclose(r_row.c, e_row.c, atol=r_row.error_c)


@check_figures_equal()
def test_plot_Shev_fit(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.Shev_fit(carbognani2019)

    test_ax = fig_test.subplots()
    pdf.plot(df=carbognani2019, ax=test_ax)

    exp_ax = fig_ref.subplots()
    exp_ax.invert_yaxis()
    exp_ax.set_title("Phase curves")
    exp_ax.set_xlabel("Phase angle")
    exp_ax.set_ylabel("V")

    def fit_y(alpha, V_lin, b, c):
        y = V_lin + c * alpha - b / (1 + alpha)
        return y

    for idx, m_row in pdf.iterrows():
        data = carbognani2019[carbognani2019["id"] == m_row.id]
        v_fit = fit_y(data.alpha, m_row.V_lin, m_row.b, m_row.c)
        exp_ax.plot(data.alpha, v_fit, "--", label=f"Fit {int(m_row.id)}")
        exp_ax.plot(
            data.alpha,
            data.v,
            marker="o",
            linestyle="None",
            label=f"Data {int(m_row.id)}",
        )
    exp_ax.legend(bbox_to_anchor=(1.05, 1))


@check_figures_equal()
def test_plot_Shev_fit_ax_None(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.Shev_fit(carbognani2019)

    exp_ax = fig_ref.subplots()
    exp_ax.invert_yaxis()
    exp_ax.set_title("Phase curves")
    exp_ax.set_xlabel("Phase angle")
    exp_ax.set_ylabel("V")

    def fit_y(alpha, V_lin, b, c):
        y = V_lin + c * alpha - b / (1 + alpha)
        return y

    for idx, m_row in pdf.iterrows():
        data = carbognani2019[carbognani2019["id"] == m_row.id]
        v_fit = fit_y(data.alpha, m_row.V_lin, m_row.b, m_row.c)
        exp_ax.plot(data.alpha, v_fit, "--", label=f"Fit {int(m_row.id)}")
        exp_ax.plot(
            data.alpha,
            data.v,
            marker="o",
            linestyle="None",
            label=f"Data {int(m_row.id)}",
        )
    exp_ax.legend(bbox_to_anchor=(1.05, 1))

    test_ax = fig_test.subplots()
    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_test):
        with mock.patch("matplotlib.pyplot.gca", return_value=test_ax):
            pdf.plot(df=carbognani2019)


@check_figures_equal()
def test_plot_DataFrame_hist_Shev(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.Shev_fit(carbognani2019)

    exp_ax = fig_ref.subplots()
    pdf.model_df.plot.hist(ax=exp_ax)

    test_ax = fig_test.subplots()
    pdf.plot.hist(ax=test_ax)


@check_figures_equal()
def test_plot_Shev_fit_curvefit(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.Shev_fit(carbognani2019)

    exp_ax = fig_ref.subplots()
    pdf.plot(df=carbognani2019, kind="curvefit", ax=exp_ax)

    test_ax = fig_test.subplots()
    pdf.plot.curvefit(df=carbognani2019, ax=test_ax)


def test_plot_invalid_plot_name_Shev(carbognani2019):
    pdf = pyedra.Shev_fit(carbognani2019)

    with pytest.raises(AttributeError):
        pdf.plot(df=carbognani2019, kind="model_df")

    with pytest.raises(AttributeError):
        pdf.plot(df=carbognani2019, kind="_foo")

    with pytest.raises(AttributeError):
        pdf.plot(df=carbognani2019, kind="foo")


@check_figures_equal()
def test_plot_Shev_fit_DataFrame_hist_by_name(
    carbognani2019, fig_test, fig_ref
):
    pdf = pyedra.Shev_fit(carbognani2019)

    exp_ax = fig_ref.subplots()
    pdf.model_df.plot(kind="hist", ax=exp_ax)

    test_ax = fig_test.subplots()
    pdf.plot(kind="hist", ax=test_ax)
