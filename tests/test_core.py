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

import pyedra

import pytest


# =============================================================================
# TESTS
# =============================================================================


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
