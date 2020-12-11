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
    exp_ax.set_title("HG - Phase curves")
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
def test_plot_HG_fit_cmap_str(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG_fit(carbognani2019)

    test_ax = fig_test.subplots()
    pdf.plot(df=carbognani2019, ax=test_ax, cmap="viridis")

    exp_ax = fig_ref.subplots()
    exp_ax.invert_yaxis()
    exp_ax.set_title("HG - Phase curves")
    exp_ax.set_xlabel("Phase angle")
    exp_ax.set_ylabel("V")

    def fit_y(alpha, H, G):
        x = alpha * np.pi / 180
        y = H - 2.5 * np.log10(
            (1 - G) * np.exp(-3.33 * np.tan(x / 2) ** 0.63)
            + G * np.exp(-1.87 * np.tan(x / 2) ** 1.22)
        )
        return y

    model_size = len(pdf.model_df)

    cmap = cm.get_cmap("viridis")
    colors = colors = cmap(np.linspace(0, 1, model_size))

    for idx, m_row in pdf.iterrows():
        data = carbognani2019[carbognani2019["id"] == m_row.id]
        v_fit = fit_y(data.alpha, m_row.H, m_row.G)

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
def test_plot_HG_fit_cmap_callable(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG_fit(carbognani2019)

    test_ax = fig_test.subplots()
    pdf.plot(df=carbognani2019, ax=test_ax, cmap=cm.get_cmap("viridis"))

    exp_ax = fig_ref.subplots()
    exp_ax.invert_yaxis()
    exp_ax.set_title("HG - Phase curves")
    exp_ax.set_xlabel("Phase angle")
    exp_ax.set_ylabel("V")

    def fit_y(alpha, H, G):
        x = alpha * np.pi / 180
        y = H - 2.5 * np.log10(
            (1 - G) * np.exp(-3.33 * np.tan(x / 2) ** 0.63)
            + G * np.exp(-1.87 * np.tan(x / 2) ** 1.22)
        )
        return y

    model_size = len(pdf.model_df)

    cmap = cm.get_cmap("viridis")
    colors = colors = cmap(np.linspace(0, 1, model_size))

    for idx, m_row in pdf.iterrows():
        data = carbognani2019[carbognani2019["id"] == m_row.id]
        v_fit = fit_y(data.alpha, m_row.H, m_row.G)

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
def test_plot_HG_fit_ax_None(carbognani2019, fig_test, fig_ref):
    pdf = pyedra.HG_fit(carbognani2019)

    test_ax = fig_test.subplots()
    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_test):
        with mock.patch("matplotlib.pyplot.gca", return_value=test_ax):
            pdf.plot(df=carbognani2019)

    fig_ref.set_size_inches(pdf.plot.DEFAULT_FIGURE_SIZE)
    exp_ax = fig_ref.subplots()
    exp_ax.invert_yaxis()
    exp_ax.set_title("HG - Phase curves")
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
