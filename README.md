# Pyedra
![logo](https://raw.githubusercontent.com/milicolazo/Pyedra/master/res/logo_bw.png)

[![Build Status](https://travis-ci.com/milicolazo/Pyedra.svg?branch=master)](https://travis-ci.com/milicolazo/Pyedra)
[![Build Status](https://github.com/milicolazo/Pyedra/actions/workflows/pyedra_ci.yml/badge.svg?branch=master)](https://github.com/milicolazo/Pyedra/actions/workflows/pyedra_ci.yml)
[![Documentation Status](https://readthedocs.org/projects/pyedra/badge/?version=latest)](https://pyedra.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/Pyedra)](https://pypi.org/project/Pyedra/)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://badge.fury.io/py/uttrs)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://tldrlegal.com/license/mit-license)
[![arXiv](https://img.shields.io/badge/arXiv-2103.06856-b31b1b.svg)](https://arxiv.org/abs/2103.06856)
[![ASCL.net](https://img.shields.io/badge/ascl-2103.008-blue.svg?colorB=262255)](https://ascl.net/2103.008)




**Pyedra** is a python library that allows you to fit three different models of asteroid phase functions to your observations.

## Motivation
Phase curves of asteroids are very important for the scientific community as they can provide information needed for diameter and albedo estimates. Given the large amount of surveys currently being carried out and planned for the next few years, we believe it is necessary to have a tool capable of processing and providing useful information for such an amount of observational data.

## Features
The input file must be a .csv file with three columns: id (mpc number), alpha (phase angle) and v (reduced magnitude in Johnson's V filter).

Pyedra currently has three functions, each of which adjusts a different phase function model.
The modules are:

- **HG_fit**: adjusts the H-G biparametric function to the data set.

- **Shev_fit**: adjusts the Shevchenko triparametric function to the data set.

- **HG1G2_fit**: adjusts the triparametric function H-G1-G2 to the data set.


In addition, the input data can be plotted with the chosen fit.


--------------------------------------------------------------------------------

## Requirements
You need Python 3.8+ to run Pyedra.

## Installation
Clone this repo and then inside the local directory execute

        $ pip install -e .
