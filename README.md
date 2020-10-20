# Pyedra
[![Build Status](https://travis-ci.com/milicolazo/Pyedra.svg?branch=master)](https://travis-ci.com/milicolazo/Pyedra)

**Pyedra** is a python library that allows you to fit three different models of asteroid phase functions to your observations.

## Motivation
Phase curves of asteroids are very important for the scientific community as they can provide information needed for diameter and albedo estimates. Given the large amount of surveys currently being carried out and planned for the next few years, we believe it is necessary to have a tool capable of processing and providing useful information for such an amount of observational data.

## Features
Pyedra currently has three functions, each of which adjusts a different phase function model. The input file must be a .csv file with three columns: id (mpc number), alpha (phase angle) and v (reduced magnitude in Johnson's V filter).
The modules are:

- **H-G_fit**: adjusts the H-G biparametric function to the data set. 

- **Shev_fit**: adjusts the Shevchenko triparametric function to the data set.

- **HG1G2_fit**: adjusts the triparametric function H-G1-G2 to the data set.

--------------------------------------------------------------------------------

## Requirements
You need Python 3.8 to run Pyedra.

## Installation
Clone this repo and then inside the local directory execute

        $ pip install -e .
