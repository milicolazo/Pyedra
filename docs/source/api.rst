pyedra API
==========

**Pyedra** currently has three functions, each of which adjusts a different phase function model.

The input file must be a .csv file with three columns: id (mpc number), alpha (phase angle) and v (reduced magnitude in Johnson's V filter).

The modules are:

- **HG_fit**: adjusts the H-G biparametric function to the data set.

- **Shev_fit**: adjusts the Shevchenko triparametric function to the data set.

- **HG1G2_fit**: adjusts the triparametric function H-G1-G2 to the data set.

In addition, the data input can be plotted with the chosen fit.

-------------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 4

Module ``pyedra.core``
----------------------

.. automodule:: pyedra.core
   :members:
   :show-inheritance:
   :member-order: bysource

Module ``pyedra.hg1g2_model``
-----------------------------

.. automodule:: pyedra.hg1g2_model
   :members:
   :show-inheritance:
   :member-order: bysource

Module ``pyedra.hg_model``
--------------------------

.. automodule:: pyedra.hg_model
   :members:
   :show-inheritance:
   :member-order: bysource

Module ``pyedra.shevchenko_model``
----------------------------------

.. automodule:: pyedra.shevchenko_model
   :members:
   :show-inheritance:
   :member-order: bysource

Module ``pyedra.datasets``
--------------------------

.. automodule:: pyedra.datasets
   :members:
   :show-inheritance:
   :member-order: bysource
