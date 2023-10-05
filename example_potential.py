#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:29:24 2023

@author: kacharov
"""

import numpy as np
from astropy import units as u

from useful_functions import gnfw_density, gnfw_potential, gnfw_params
from solve_poisson import get_potential

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

# Define 3D density parameters to test a cored and a cusped gNFW profile.
M200 = 1e11 *u.Msun # Virial mass for a gNFW profile, Msun
cc = 10**1.5  # Concentration for a gNFW

# get the gNFW scale density and radius
rho_s_nfw, r_s_nfw = gnfw_params(M200, cc, gamma=1)
rho_s_core, r_s_core = gnfw_params(M200, cc, gamma=0)

# at the moment the potential routines require unitless values
# it is assumed that r is in pc, density in Msun/pc**3
rho_s_nfw = rho_s_nfw.value
r_s_nfw = r_s_nfw.value
rho_s_core = rho_s_core.value
r_s_core = r_s_core.value

# Define the radial coordinate
#r = np.logspace(0, 4, 1000)
r = np.linspace(1, 1e4, 1000)

# Get the analitic potentials
phi_analytic_nfw = gnfw_potential(r, rho_s_nfw, r_s_nfw, 1)
phi_analytic_core = gnfw_potential(r, rho_s_core, r_s_core, 0)

# Solve Poisson and get the numerical potentials
phi_numerical_nfw = get_potential(r, gnfw_density, rho_s_nfw, r_s_nfw, 1)
phi_numerical_core = get_potential(r, gnfw_density, rho_s_core, r_s_core, 0)

# Plot the results to compare
plt.figure(figsize=(10, 7))
plt.plot(r, phi_numerical_nfw, label="Numerical Solution")
plt.plot(r, phi_analytic_nfw, '--', label="Analytic Solution")
#plt.xscale("log")
plt.xlabel("Radius (pc)")
plt.ylabel("$\Phi$ (km^2 s^-2)")
plt.title("$\Phi$ for NFW profile (analytic vs numerical)")
plt.legend()
plt.grid(True)
plt.show()

# Plot the results to compare
plt.figure(figsize=(10, 7))
plt.plot(r, phi_numerical_core, label="Numerical Solution")
plt.plot(r, phi_analytic_core, '--', label="Analytic Solution")
#plt.xscale("log")
plt.xlabel("Radius (pc)")
plt.ylabel("$\Phi$ (km^2 s^-2)")
plt.title("$\Phi$ for Cored profile (analytic vs numerical)")
plt.legend()
plt.grid(True)
plt.show()
