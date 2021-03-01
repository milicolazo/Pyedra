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

    def __getattr__(self, y):
        """getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
        return getattr(self.pdf.model_df.plot, y)


# ============================================================================
# FUNCTIONS
# ============================================================================


def obs_counter(df, obs, idc="id", alphac="alpha"):
    """Count the observations. A minimum is needed to the fits.

    Parameters
    ----------
    df: ``pandas.DataFrame``
        The dataframe must with the values
    idc : ``str``, optional (default=id)
        Column with the mpc number of the asteroids.
    alphac : ``str``, optional (default=alpha)
        Column with the phase angle of the asteroids.
    obs: ``int``
        Minimum number of observations needed to perform the fit.

    Return
    ------
    out: ndarray
        Numpy array containing the asteroids whose number of
        observations is less than obs.
    """
    df_cnt = df.groupby(idc).count()
    lt_idx = df_cnt[df_cnt[alphac] < obs].index
    return lt_idx.to_numpy()


def merge_obs(
    obs_a,
    obs_b,
    idc_a="id",
    idc_b="id",
    alphac_a="alpha",
    alphac_b="alpha",
    magc_a="v",
    magc_b="v",
    **kwargs,
):
    """Merge two dataframes with observations.

    Sources whose id is not present in `obs_a` are discarded.

    The function concantenates two dataframes (``obs_a`` and ``obs_b``), and
    assumes that the columns ``idc_a``, ``alphac_a`` and ``magc_a`` from
    ``obs_a`` are equivalent to ``idc_b``, ``alphac_b`` and ``magc_b``
    from a dataframe ``obs_b``.
    The resulting dataframe uses the names of
    ``obs_a`` for those three columns and places them at the start, and all
    other columns of both dataframes combined with the same behavior of
    ``pandas.concat``.

    Parameters
    ----------
    obs_a: ``pandas.DataFrame``
        The dataframe must with the observations.
    obs_b: ``pandas.DataFrame``
        The dataframe must with the observations to be concatenated
        to ``obs_a``.
    idc_a : ``str``, optional (default=id)
        Column with the mpc number of the asteroids of the ``obs_a`` dataframe.
    idc_b : ``str``, optional (default=id)
        Column with the mpc number of the asteroids of the ``obs_b`` dataframe.
    alphac_a : ``str``, optional (default=alpha)
        Column with the phase angle of the asteroids of the ``obs_a``
        dataframe.
    alphac_b : ``str``, optional (default=alpha)
        Column with the phase angle of the asteroids of the ``obs_b``
        dataframe.
    magc_a : ``str``, optional (default=v)
        Column with the magnitude of the ``obs_a`` dataframe.
        The default 'v' value is reference to the reduced magnitude in
        Johnson's V filter.
    magc_b : ``str``, optional (default=v)
        Column with the magnitude of the ``obs_b`` dataframe.
        The default 'v' value is reference to the reduced magnitude in
        Johnson's V filter.
    kwargs: ``dict`` or ``None`` (optional)
        The parameters to send to the subjacent ``pandas.concat`` function.

    Return
    ------
    ``pd.DataFrame`` :
        Merged dataframes.

    """
    # set the order of the first 3 columns of the obs_a and the merged df
    columns_a = [idc_a, alphac_a, magc_a] + [
        c for c in obs_a.columns if c not in [idc_a, alphac_a, magc_a]
    ]
    obs_a = obs_a[columns_a]

    # retrieve only tne ids of the first datasets from the secondone
    ids = obs_a[idc_a].unique()
    obs_b = obs_b[obs_b[idc_b].isin(ids)].copy()

    # rename the columns of obs_b according to obs_a names
    map_b_col_names = {idc_b: idc_a, alphac_b: alphac_a, magc_b: magc_a}
    columns_b = [map_b_col_names.get(c, c) for c in obs_b.columns]
    obs_b.columns = columns_b

    # set the default configuration of pd.concat
    kwargs.setdefault("ignore_index", True)

    # merge and return
    merged = pd.concat([obs_a, obs_b], **kwargs)
    return merged
