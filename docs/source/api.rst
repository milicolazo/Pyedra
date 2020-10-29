API Module
==========

**Pyedra** currently has three functions, each of which adjusts a different phase function model. The input file must be a .csv file
with three columns: id (mpc number), alpha (phase angle) and v (reduced magnitude in Johnson's V filter). The modules are:

- **H-G_fit**: adjusts the H-G biparametric function to the data set. 

- **Shev_fit**: adjusts the Shevchenko triparametric function to the data set.

- **HG1G2_fit**: adjusts the triparametric function H-G1-G2 to the data set. 

----------------------------------------

Pyedra
------

.. toctree::
   :maxdepth: 4

   pyedra.core
