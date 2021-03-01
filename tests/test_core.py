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


def test_repr(carbognani2019):
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


@pytest.mark.parametrize("idc", ["id", "num"])
@pytest.mark.parametrize("alphac", ["alpha", "alph"])
def test_obs_counter_other_column_names(carbognani2019, idc, alphac):
    expected = pyedra.obs_counter(carbognani2019, 8)

    carbognani2019.columns = [idc, alphac, "xx"]
    result = pyedra.obs_counter(carbognani2019, 8, idc=idc, alphac=alphac)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("idc_a", ["id", "num"])
@pytest.mark.parametrize("idc_b", ["id", "num"])
@pytest.mark.parametrize("alphac_a", ["alpha", "alph"])
@pytest.mark.parametrize("alphac_b", ["alpha", "alph"])
@pytest.mark.parametrize("magc_a", ["v", "i"])
@pytest.mark.parametrize("magc_b", ["v", "i"])
@pytest.mark.parametrize("extrac_a", [0, 5])
@pytest.mark.parametrize("extrac_b", [0, 5])
@pytest.mark.parametrize("extra_obs_b", [0, 10])
def test_obs_merge(
    carbognani2019,
    idc_a,
    alphac_a,
    magc_a,
    idc_b,
    alphac_b,
    magc_b,
    extrac_a,
    extrac_b,
    extra_obs_b,
):
    obs_a = carbognani2019

    # first calculate create the extra columns
    extrac_a = [f"col_a_{idx}" for idx in range(extrac_a)]
    extrac_b = [f"col_b_{idx}" for idx in range(extrac_b)]

    obs_a.columns = [idc_a, alphac_a, magc_a]
    obs_a_count = obs_a.groupby(idc_a).count()[magc_a].to_dict()

    obs_b_count = {
        oid: np.random.randint(0, 100) for oid in obs_a[idc_a].unique()
    }
    rows = []
    for oid, new_obs in obs_b_count.items():
        for _ in range(new_obs):
            rows.append({idc_b: oid, alphac_b: -100, magc_b: -100})

    # agregamos filas con id sin sentido que no tienen que llegar al final
    max_id = np.max(obs_a[idc_a].unique())
    for idx in range(extra_obs_b):
        oid = max_id + idx + 1
        rows.append({idc_b: oid, alphac_b: -100, magc_b: -100})

    # creamos el dataframe
    obs_b = pd.DataFrame(rows)

    # add the extra columns
    obs_a_len = len(obs_a)
    for ec in extrac_a:
        obs_a = obs_a.assign(**{ec: np.random.random(size=obs_a_len)})

    obs_b_len = len(obs_b)
    for ec in extrac_b:
        obs_b = obs_b.assign(**{ec: np.random.random(size=obs_b_len)})

    merged = pyedra.merge_obs(
        obs_a=obs_a,
        obs_b=obs_b,
        idc_a=idc_a,
        alphac_a=alphac_a,
        magc_a=magc_a,
        idc_b=idc_b,
        alphac_b=alphac_b,
        magc_b=magc_b,
    )

    merged_count = merged.groupby(idc_a).count()[magc_a].to_dict()
    for oid, mcount in merged_count.items():
        assert mcount == obs_a_count[oid] + obs_b_count[oid]

    expected_cols = [idc_a, alphac_a, magc_a] + list(extrac_a) + list(extrac_b)
    np.testing.assert_array_equal(merged.columns, expected_cols)

    assert sorted(merged[idc_a].unique()) == sorted(obs_a[idc_a].unique())
