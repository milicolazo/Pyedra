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

import attr

import numpy as np

import pandas as pd

import pyedra

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_getitem(carbognani2019, expected_filter):
    pdf = pyedra.HG_fit(carbognani2019).model_df

    filt = pdf[pdf["id"] == 85]

    np.testing.assert_array_equal(filt.id, expected_filter.id)
    np.testing.assert_array_almost_equal(filt.H, expected_filter.H)
    np.testing.assert_array_almost_equal(
        filt.error_H,
        expected_filter.error_H,
    )
    np.testing.assert_array_almost_equal(filt.G, expected_filter.G)
    np.testing.assert_array_almost_equal(filt.error_G, expected_filter.error_G)
    np.testing.assert_array_almost_equal(filt.R, expected_filter.R)


def test_rep(carbognani2019):
    pdf = pyedra.HG_fit(carbognani2019)

    with pd.option_context("display.show_dimensions", False):
        df_body = repr(pdf.model_df).splitlines()
    df_dim = list(pdf.model_df.shape)
    sdf_dim = f"{df_dim[0]} rows x {df_dim[1]} columns"

    fotter = f"\nPyedraFitDataFrame - {sdf_dim}"
    expected = "\n".join(df_body + [fotter])

    assert repr(pdf) == expected


def test_repr_html(carbognani2019):
    pdf = pyedra.HG_fit(carbognani2019)

    ad_id = id(pdf)
    with pd.option_context("display.show_dimensions", False):
        df_html = pdf.model_df._repr_html_()
    rows = f"{pdf.model_df.shape[0]} rows"
    columns = f"{pdf.model_df.shape[1]} columns"
    footer = f"PyedraFitDataFrame - {rows} x {columns}"
    parts = [
        f'<div class="pyedra-data-container" id={ad_id}>',
        df_html,
        footer,
        "</div>",
    ]

    expected = "".join(parts)

    assert pdf._repr_html_() == expected


def test_dir(carbognani2019):
    pdf = pyedra.HG_fit(carbognani2019)

    for i in dir(pdf):
        assert hasattr(pdf, i)


def test_metadata():
    original = {"a": 1, "b": 2}

    metadata = pyedra.MetaData(original)
    assert metadata.a == metadata["a"] == original["a"]
    assert metadata.b == metadata["b"] == original["b"]

    with pytest.raises(KeyError):
        metadata["foo"]
    with pytest.raises(KeyError):
        metadata["foo"]

    assert repr(metadata) == f"Metadata({repr(original)})"
    assert len(metadata) == len(original)
    assert list(iter(metadata)) == list(iter(original))

    with pytest.raises(TypeError):
        metadata["foo"] = 1

    with pytest.raises(attr.exceptions.FrozenInstanceError):
        metadata.foo = 1


def test_obs_counter(carbognani2019):

    result = pyedra.obs_counter(carbognani2019, 8)

    expected = [85, 208, 306, 313, 338, 522]

    np.testing.assert_array_almost_equal(result, expected, 6)
