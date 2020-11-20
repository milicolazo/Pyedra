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

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pandas as pd

import pyedra.datasets

import pytest

import scipy
import scipy.interpolate

# =============================================================================
# CONSTANTS
# =============================================================================


@pytest.fixture(scope="session")
def carbognani2019():
    return pyedra.datasets.load_carbognani2019()


@pytest.fixture
def bad_data():
    return pd.DataFrame({"id": {0: 85}, "alpha": {0: 5}, "v": {0: 8}})


# =============================================================================
# TESTS
# =============================================================================
def test_obs_counter(carbognani2019):

    result = pyedra.obs_counter(carbognani2019, 8)

    expected = [85, 208, 306, 313, 338, 522]

    np.testing.assert_array_almost_equal(result, expected, 6)


# ------------------------------------------------------------------------------
# H, G
# ------------------------------------------------------------------------------
def test_raises_HG(bad_data):
    with pytest.raises(ValueError):
        pyedra.HG_fit(bad_data)


def test_PyedraFitDataFrame_HG(carbognani2019):
    pdf = pyedra.HG_fit(carbognani2019)

    np.testing.assert_array_equal(pdf.id, pdf.model_df.id)
    np.testing.assert_array_equal(pdf.H, pdf.model_df.H)
    np.testing.assert_array_equal(pdf.G, pdf.model_df.G)


def test_HG_fit(carbognani2019):

    result = pyedra.HG_fit(carbognani2019).model_df

    expected = pd.DataFrame(
        {
            "id": {0: 85, 1: 208, 2: 236, 3: 306, 4: 313, 5: 338, 6: 522},
            "H": {0: 7.52, 1: 9.2, 2: 8.0, 3: 8.79, 4: 8.87, 5: 8.51, 6: 9.0},
            "G": {
                0: 0.0753,
                1: 0.2761,
                2: 0.0715,
                3: 0.2891,
                4: 0.1954,
                5: -0.0812,
                6: 0.1411,
            },
        }
    )

    for idx, e_row in expected.iterrows():
        r_row = result[result.id == e_row.id].iloc[0]
        np.testing.assert_array_equal(r_row.id, e_row.id)
        np.testing.assert_allclose(r_row.H, e_row.H, atol=r_row.error_H)
        np.testing.assert_allclose(r_row.G, e_row.G, atol=r_row.error_G)


@check_figures_equal()
def test_plot_HG_fit(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG_fit(carbognani2019)

    test_ax = fig_test.subplots()
    pdf.plot(df=carbognani2019, ax=test_ax)

    exp_ax = fig_ref.subplots()
    exp_ax.invert_yaxis()
    exp_ax.set_title("Phase curves")
    exp_ax.set_xlabel("Phase angle")
    exp_ax.set_ylabel("V")

    def fit_y(alpha, H, G):
        x = alpha * np.pi / 180
        y = H - 2.5 * np.log10(
            (1 - G) * np.exp(-3.33 * np.tan(x / 2) ** 0.63)
            + G * np.exp(-1.87 * np.tan(x / 2) ** 1.22)
        )
        return y

    for idx, m_row in pdf.iterrows():
        data = carbognani2019[carbognani2019["id"] == m_row.id]
        v_fit = fit_y(data.alpha, m_row.H, m_row.G)
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
def test_plot_DataFrame_hist_HG(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG_fit(carbognani2019)

    exp_ax = fig_ref.subplots()
    pdf.model_df.plot.hist(ax=exp_ax)

    test_ax = fig_test.subplots()
    pdf.plot.hist(ax=test_ax)


@check_figures_equal()
def test_plot_HG_fit_curvefit(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG_fit(carbognani2019)

    exp_ax = fig_ref.subplots()
    pdf.plot(df=carbognani2019, kind="curvefit", ax=exp_ax)

    test_ax = fig_test.subplots()
    pdf.plot.curvefit(df=carbognani2019, ax=test_ax)


def test_plot_invalid_plot_name_HG(carbognani2019):
    pdf = pyedra.HG_fit(carbognani2019)

    with pytest.raises(AttributeError):
        pdf.plot(df=carbognani2019, kind="model_df")

    with pytest.raises(AttributeError):
        pdf.plot(df=carbognani2019, kind="_foo")

    with pytest.raises(AttributeError):
        pdf.plot(df=carbognani2019, kind="foo")


@check_figures_equal()
def test_plot_HG_fit_DataFrame_hist_by_name(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG_fit(carbognani2019)

    exp_ax = fig_ref.subplots()
    pdf.model_df.plot(kind="hist", ax=exp_ax)

    test_ax = fig_test.subplots()
    pdf.plot(kind="hist", ax=test_ax)


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


# -----------------------------------------------------------------------------
# H, G1, G2
# -----------------------------------------------------------------------------
def test_raises_HG1G2(bad_data):
    with pytest.raises(ValueError):
        pyedra.HG1G2_fit(bad_data)


def test_PyedraFitDataFrame_HG1G2(carbognani2019):
    pdf = pyedra.HG1G2_fit(carbognani2019)

    np.testing.assert_array_equal(pdf.id, pdf.model_df.id)
    np.testing.assert_array_equal(pdf.H12, pdf.model_df.H12)
    np.testing.assert_array_equal(pdf.G1, pdf.model_df.G1)
    np.testing.assert_array_equal(pdf.G2, pdf.model_df.G2)


def test_HG1G2_fit(carbognani2019):

    result = pyedra.HG1G2_fit(carbognani2019).model_df

    expected = pd.DataFrame(
        {
            "id": {0: 85, 1: 208, 2: 236, 3: 306, 4: 313, 5: 338, 6: 522},
            "H12": {
                0: 7.41,
                1: 8.92,
                2: 7.82,
                3: 8.04,
                4: 8.88,
                5: 8.41,
                6: 9.07,
            },
            "G1": {
                0: 0.3358,
                1: -0.3116,
                2: 0.1155,
                3: -0.1309,
                4: 0.624,
                5: 0.5607,
                6: 0.7302,
            },
            "G2": {
                0: 0.2147,
                1: 0.6598,
                2: 0.3436,
                3: 0.3624,
                4: 0.151,
                5: -0.0037,
                6: 0.0879,
            },
        }
    )

    for idx, e_row in expected.iterrows():
        r_row = result[result.id == e_row.id].iloc[0]
        np.testing.assert_array_equal(r_row.id, e_row.id)
        np.testing.assert_allclose(r_row.H12, e_row.H12, atol=r_row.error_H12)
        np.testing.assert_allclose(r_row.G1, e_row.G1, atol=r_row.error_G1)
        np.testing.assert_allclose(r_row.G2, e_row.G2, atol=r_row.error_G2)


@check_figures_equal()
def test_plot_HG1G2_fit(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG1G2_fit(carbognani2019)

    penttila2016 = pyedra.datasets.load_penttila2016()

    alphap = penttila2016["alpha"].to_numpy()
    phi1 = penttila2016["phi1"].to_numpy()
    phi2 = penttila2016["phi2"].to_numpy()
    phi3 = penttila2016["phi3"].to_numpy()

    y_interp1 = scipy.interpolate.interp1d(alphap, phi1)
    y_interp2 = scipy.interpolate.interp1d(alphap, phi2)
    y_interp3 = scipy.interpolate.interp1d(alphap, phi3)

    test_ax = fig_test.subplots()
    pdf.plot(df=carbognani2019, ax=test_ax)

    exp_ax = fig_ref.subplots()
    exp_ax.invert_yaxis()
    exp_ax.set_title("Phase curves")
    exp_ax.set_xlabel("Phase angle")
    exp_ax.set_ylabel("V")

    for idx, m_row in pdf.iterrows():

        data = carbognani2019[carbognani2019["id"] == m_row.id]

        def fit_y(d, e, f):
            y = d - 2.5 * np.log10(e * fi1 + f * fi2 + (1 - e - f) * fi3)
            return y

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

        v_fit = fit_y(m_row.H12, m_row.G1, m_row.G2)
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
def test_plot_DataFrame_hist_HG1G2(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG1G2_fit(carbognani2019)

    exp_ax = fig_ref.subplots()
    pdf.model_df.plot.hist(ax=exp_ax)

    test_ax = fig_test.subplots()
    pdf.plot.hist(ax=test_ax)


@check_figures_equal()
def test_plot_HG1G2_curvefit(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG1G2_fit(carbognani2019)

    exp_ax = fig_ref.subplots()
    pdf.plot(df=carbognani2019, kind="curvefit", ax=exp_ax)

    test_ax = fig_test.subplots()
    pdf.plot.curvefit(df=carbognani2019, ax=test_ax)


def test_plot_invalid_plot_name_HG1G2(carbognani2019):
    pdf = pyedra.HG1G2_fit(carbognani2019)

    with pytest.raises(AttributeError):
        pdf.plot(df=carbognani2019, kind="model_df")

    with pytest.raises(AttributeError):
        pdf.plot(df=carbognani2019, kind="_foo")

    with pytest.raises(AttributeError):
        pdf.plot(df=carbognani2019, kind="foo")


@check_figures_equal()
def test_plot_HG1G2_fit_DataFrame_hist_by_name(
    carbognani2019, fig_test, fig_ref
):
    pdf = pyedra.HG1G2_fit(carbognani2019)

    exp_ax = fig_ref.subplots()
    pdf.model_df.plot(kind="hist", ax=exp_ax)

    test_ax = fig_test.subplots()
    pdf.plot(kind="hist", ax=test_ax)
