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

from matplotlib import cm
from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pandas as pd

import pyedra.datasets

import pytest

import scipy
import scipy.interpolate

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
            "observations": {
                0: 7,
                1: 7,
                2: 8,
                3: 7,
                4: 6,
                5: 5,
                6: 7,
            },
        }
    )

    for idx, e_row in expected.iterrows():
        r_row = result[result.id == e_row.id].iloc[0]
        np.testing.assert_array_equal(r_row.id, e_row.id)
        np.testing.assert_array_equal(r_row.observations, e_row.observations)
        np.testing.assert_allclose(r_row.H12, e_row.H12, atol=r_row.error_H12)
        np.testing.assert_allclose(r_row.G1, e_row.G1, atol=r_row.error_G1)
        np.testing.assert_allclose(r_row.G2, e_row.G2, atol=r_row.error_G2)


@pytest.mark.parametrize("idc", ["id", "num"])
@pytest.mark.parametrize("alphac", ["alpha", "alph"])
@pytest.mark.parametrize("magc", ["v", "i"])
def test_HG1G2_other_column_names(carbognani2019, idc, alphac, magc):

    expected = pyedra.HG1G2_fit(carbognani2019).model_df

    carbognani2019.columns = [idc, alphac, magc]
    result = pyedra.HG1G2_fit(
        carbognani2019, idc=idc, alphac=alphac, magc=magc
    ).model_df

    pd.testing.assert_frame_equal(result, expected)


@check_figures_equal()
def test_plot_HG1G2_fit(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG1G2_fit(carbognani2019)

    test_ax = fig_test.subplots()
    pdf.plot(df=carbognani2019, ax=test_ax)

    penttila2016 = pyedra.datasets.load_penttila2016()

    alphap = penttila2016["alpha"].to_numpy()
    phi1 = penttila2016["phi1"].to_numpy()
    phi2 = penttila2016["phi2"].to_numpy()
    phi3 = penttila2016["phi3"].to_numpy()

    y_interp1 = scipy.interpolate.interp1d(alphap, phi1)
    y_interp2 = scipy.interpolate.interp1d(alphap, phi2)
    y_interp3 = scipy.interpolate.interp1d(alphap, phi3)

    exp_ax = fig_ref.subplots()
    exp_ax.invert_yaxis()
    exp_ax.set_title("HG1G2 - Phase curves")
    exp_ax.set_xlabel("Phase angle")
    exp_ax.set_ylabel("V")

    def fit_y(d, e, f):
        y = d - 2.5 * np.log10(e * fi1 + f * fi2 + (1 - e - f) * fi3)
        return y

    for idx, m_row in pdf.iterrows():

        data = carbognani2019[carbognani2019["id"] == m_row.id]

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
        line = exp_ax.plot(
            data.alpha, v_fit, "--", label=f"Fit #{int(m_row.id)}", alpha=0.5
        )
        exp_ax.plot(
            data.alpha,
            data.v,
            marker="o",
            color=line[0].get_color(),
            linestyle="None",
            label=f"Data #{int(m_row.id)}",
        )

    handles, labels = exp_ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    exp_ax.legend(handles, labels, ncol=2, loc="best")


@pytest.mark.parametrize("idc", ["id", "num"])
@pytest.mark.parametrize("alphac", ["alpha", "alph"])
@pytest.mark.parametrize("magc", ["v", "i"])
@check_figures_equal()
def test_plot_HG1G2_fit_other_column_names(
    carbognani2019, fig_test, fig_ref, idc, alphac, magc
):
    pdf = pyedra.HG1G2_fit(carbognani2019)

    exp_ax = fig_ref.subplots()
    pdf.plot(df=carbognani2019, ax=exp_ax)

    carbognani2019.columns = [idc, alphac, magc]
    test_ax = fig_test.subplots()
    pdf.plot(df=carbognani2019, idc=idc, alphac=alphac, magc=magc, ax=test_ax)
    test_ax.set_ylabel("V")


@check_figures_equal()
def test_plot_HG1G2_fit_cmap_str(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG1G2_fit(carbognani2019)

    test_ax = fig_test.subplots()
    pdf.plot(df=carbognani2019, ax=test_ax, cmap="viridis")

    penttila2016 = pyedra.datasets.load_penttila2016()

    alphap = penttila2016["alpha"].to_numpy()
    phi1 = penttila2016["phi1"].to_numpy()
    phi2 = penttila2016["phi2"].to_numpy()
    phi3 = penttila2016["phi3"].to_numpy()

    y_interp1 = scipy.interpolate.interp1d(alphap, phi1)
    y_interp2 = scipy.interpolate.interp1d(alphap, phi2)
    y_interp3 = scipy.interpolate.interp1d(alphap, phi3)

    exp_ax = fig_ref.subplots()
    exp_ax.invert_yaxis()
    exp_ax.set_title("HG1G2 - Phase curves")
    exp_ax.set_xlabel("Phase angle")
    exp_ax.set_ylabel("V")

    def fit_y(d, e, f):
        y = d - 2.5 * np.log10(e * fi1 + f * fi2 + (1 - e - f) * fi3)
        return y

    model_size = len(pdf.model_df)

    cmap = cm.get_cmap("viridis")
    colors = colors = cmap(np.linspace(0, 1, model_size))

    for idx, m_row in pdf.iterrows():

        data = carbognani2019[carbognani2019["id"] == m_row.id]

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
        line = exp_ax.plot(
            data.alpha,
            v_fit,
            "--",
            label=f"Fit #{int(m_row.id)}",
            alpha=0.5,
            color=colors[idx],
        )
        exp_ax.plot(
            data.alpha,
            data.v,
            marker="o",
            color=line[0].get_color(),
            linestyle="None",
            label=f"Data #{int(m_row.id)}",
        )

    handles, labels = exp_ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    exp_ax.legend(handles, labels, ncol=2, loc="best")


@check_figures_equal()
def test_plot_HG1G2_fit_cmap_callable(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG1G2_fit(carbognani2019)

    test_ax = fig_test.subplots()
    pdf.plot(df=carbognani2019, ax=test_ax, cmap=cm.get_cmap("viridis"))

    penttila2016 = pyedra.datasets.load_penttila2016()

    alphap = penttila2016["alpha"].to_numpy()
    phi1 = penttila2016["phi1"].to_numpy()
    phi2 = penttila2016["phi2"].to_numpy()
    phi3 = penttila2016["phi3"].to_numpy()

    y_interp1 = scipy.interpolate.interp1d(alphap, phi1)
    y_interp2 = scipy.interpolate.interp1d(alphap, phi2)
    y_interp3 = scipy.interpolate.interp1d(alphap, phi3)

    exp_ax = fig_ref.subplots()
    exp_ax.invert_yaxis()
    exp_ax.set_title("HG1G2 - Phase curves")
    exp_ax.set_xlabel("Phase angle")
    exp_ax.set_ylabel("V")

    def fit_y(d, e, f):
        y = d - 2.5 * np.log10(e * fi1 + f * fi2 + (1 - e - f) * fi3)
        return y

    model_size = len(pdf.model_df)

    cmap = cm.get_cmap("viridis")
    colors = colors = cmap(np.linspace(0, 1, model_size))

    for idx, m_row in pdf.iterrows():

        data = carbognani2019[carbognani2019["id"] == m_row.id]

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
        line = exp_ax.plot(
            data.alpha,
            v_fit,
            "--",
            label=f"Fit #{int(m_row.id)}",
            alpha=0.5,
            color=colors[idx],
        )
        exp_ax.plot(
            data.alpha,
            data.v,
            marker="o",
            color=line[0].get_color(),
            linestyle="None",
            label=f"Data #{int(m_row.id)}",
        )

    handles, labels = exp_ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    exp_ax.legend(handles, labels, ncol=2, loc="best")


@check_figures_equal()
def test_plot_HG1G2_fit_ax_None(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG1G2_fit(carbognani2019)

    test_ax = fig_test.subplots()
    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_test):
        with mock.patch("matplotlib.pyplot.gca", return_value=test_ax):
            pdf.plot(df=carbognani2019)

    penttila2016 = pyedra.datasets.load_penttila2016()

    alphap = penttila2016["alpha"].to_numpy()
    phi1 = penttila2016["phi1"].to_numpy()
    phi2 = penttila2016["phi2"].to_numpy()
    phi3 = penttila2016["phi3"].to_numpy()

    y_interp1 = scipy.interpolate.interp1d(alphap, phi1)
    y_interp2 = scipy.interpolate.interp1d(alphap, phi2)
    y_interp3 = scipy.interpolate.interp1d(alphap, phi3)

    fig_ref.set_size_inches(pdf.plot.DEFAULT_FIGURE_SIZE)
    exp_ax = fig_ref.subplots()
    exp_ax.invert_yaxis()
    exp_ax.set_title("HG1G2 - Phase curves")
    exp_ax.set_xlabel("Phase angle")
    exp_ax.set_ylabel("V")

    def fit_y(d, e, f):
        y = d - 2.5 * np.log10(e * fi1 + f * fi2 + (1 - e - f) * fi3)
        return y

    for idx, m_row in pdf.iterrows():

        data = carbognani2019[carbognani2019["id"] == m_row.id]

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
        v_fit = fit_y(m_row.H12, m_row.G1, m_row.G2)
        line = exp_ax.plot(
            data.alpha, v_fit, "--", label=f"Fit #{int(m_row.id)}", alpha=0.5
        )
        exp_ax.plot(
            data.alpha,
            data.v,
            marker="o",
            color=line[0].get_color(),
            linestyle="None",
            label=f"Data #{int(m_row.id)}",
        )

    handles, labels = exp_ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    exp_ax.legend(handles, labels, ncol=2, loc="best")


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
