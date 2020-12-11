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

import abc
from collections.abc import Mapping

import attr

import pandas as pd

# ============================================================================
# CLASSES
# ============================================================================


@attr.s(frozen=True, repr=False)
class MetaData(Mapping):
    """Implements an inmutable dict-like to store the metadata.

    Also provides attribute like access to the keys.

    Example
    -------
    >>> metadata = MetaData({"a": 1, "b": 2})
    >>> metadata.a
    1

    >>> metadata["a"]
    1
    """

    _data = attr.ib(converter=dict, factory=dict)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        return f"Metadata({repr(self._data)})"

    def __getitem__(self, k):
        """x[k] <=> x.__getitem__(k)."""
        return self._data[k]

    def __iter__(self):
        """iter(x) <=> x.__iter__()."""
        return iter(self._data)

    def __len__(self):
        """len(x) <=> x.__len__()."""
        return len(self._data)

    def __getattr__(self, a):
        """getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
        return self[a]


@attr.s(frozen=True, repr=False)
class PyedraFitDataFrame:
    """Initialize a dataframe model_df to which we can apply function a."""

    model = attr.ib(validator=attr.validators.instance_of(str))
    model_df = attr.ib(validator=attr.validators.instance_of(pd.DataFrame))
    plot_cls = attr.ib()

    metadata = attr.ib(factory=MetaData, converter=MetaData)
    plot = attr.ib(init=False)

    @plot.default
    def _plot_default(self):
        return self.plot_cls(self)

    def __getitem__(self, slice):
        """x[y] <==> x.__getitem__(y)."""
        sliced = self.model_df.__getitem__(slice)
        return PyedraFitDataFrame(
            model=self.model,
            model_df=sliced,
            plot_cls=self.plot_cls,
            metadata=dict(self.metadata),
        )

    def __dir__(self):
        """dir(pdf) <==> pdf.__dir__()."""
        return super().__dir__() + dir(self.model_df)

    def __getattr__(self, a):
        """getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
        return getattr(self.model_df, a)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        with pd.option_context("display.show_dimensions", False):
            df_body = repr(self.model_df).splitlines()
        df_dim = list(self.model_df.shape)
        sdf_dim = f"{df_dim[0]} rows x {df_dim[1]} columns"

        fotter = f"\nPyedraFitDataFrame - {sdf_dim}"
        pyedra_data_repr = "\n".join(df_body + [fotter])
        return pyedra_data_repr

    def _repr_html_(self):
        ad_id = id(self)

        with pd.option_context("display.show_dimensions", False):
            df_html = self.model_df._repr_html_()

        rows = f"{self.model_df.shape[0]} rows"
        columns = f"{self.model_df.shape[1]} columns"

        footer = f"PyedraFitDataFrame - {rows} x {columns}"

        parts = [
            f'<div class="pyedra-data-container" id={ad_id}>',
            df_html,
            footer,
            "</div>",
        ]

        html = "".join(parts)
        return html


@attr.s(frozen=True)
class BasePlot(abc.ABC):
    """Plots for HG fit."""

    # this is the default size of any plot.
    DEFAULT_FIGURE_SIZE = 10, 6

    pdf = attr.ib()

    @abc.abstractproperty
    def default_plot_kind(self):
        """Return the default plot to be rendered."""

    def __call__(self, kind=None, **kwargs):
        """``plot() <==> plot.__call__()``."""
        kind = self.default_plot_kind if kind is None else kind

        if kind.startswith("_"):
            raise AttributeError(f"Ivalid plot method '{kind}'")

        method = getattr(self, kind)

        if not callable(method):
            raise AttributeError(f"Ivalid plot method '{kind}'")

        return method(**kwargs)

    def __getattr__(self, kind):
        """getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
        return getattr(self.pdf.model_df.plot, kind)


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
