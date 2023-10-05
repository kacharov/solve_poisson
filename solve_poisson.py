#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:25:32 2023

@author: kacharov
"""

import numpy as np
from scipy.integrate import odeint


# Constants
G = 4.3009172706e-3  # Gravitational constant, pc Msun^-1 (km/s)^2

# Poisson's equation, discretized


def solve_poisson(y, r, rho_func, *rho_params):
    """
    Computes the derivatives for the Poisson's equation, given the current state
    of the system and the 3D density function.

    The Poisson's equation is solved in its first order, equivalent form:
    1/r**2 * d/dr(r**2 * dΦ/dr) = 4πGρ(r)
    which is recast as:
    dΦ/dr = p
    dp/dr = 4πGρ(r) - 2p / r

    Parameters
    ----------
    y : array_like, shape (2,)
        The current state of the system. y[0] = Φ is the gravitational potential 
        and y[1] = p is the first derivative of the potential.

    r : float
        The current radial distance.

    rho_func : callable
        The 3D density function. It must be a function rho_func(r, *rho_params)
        that takes the radial distance and additional parameters as inputs and 
        returns the mass density.

    *rho_params : sequence of float
        Additional parameters to `rho_func`.

    Returns
    -------
    dphi_dr : float
        The first derivative of the gravitational potential (i.e., y[1]).

    dp_dr : float
        The derivative of y[1]. This will be zero if `r` is zero.

    Notes
    -----
    - This function is intended to be used with scipy.integrate.odeint, which
      numerically integrates systems of first-order 
      ordinary differential equations (ODEs).

    - The density function and its parameters are input arguments to this 
      function. This allows the Poisson's equation to be solved for 
      different density profiles by simply changing these input arguments.

    - If r is zero, then dp_dr is set to zero to avoid the singularity.
    """

    phi, p = y
    rho = rho_func(r, *rho_params)
    dphi_dr = p
    dp_dr = 4 * np.pi * G * rho - 2 * p / r if r != 0 else 0
    return [dphi_dr, dp_dr]


def get_potential(r, rho_func, *rho_params):
    """
    Computes the numerical solution to Poisson's equation given a density
    profile.

    This function uses the scipy.integrate.odeint solver to integrate
    Poisson's equation for a given density profile.
    The function evaluates the potential for an array of radial distances `r`.

    Parameters
    ----------
    r : array_like
        Array of radial distances at which to compute the potential.

    rho_func : callable
        The mass density function. It must be a function
        rho_func(r, *rho_params) that takes the radial distance and additional
        parameters as inputs and returns the mass density.

    *rho_params : sequence of float
        Additional parameters to `rho_func`.

    Returns
    -------
    phi_numerical : numpy.ndarray
        The gravitational potential corresponding to the given density profile,
        computed at each radius in r. The potential is adjusted such that it
        reaches 0 at r_max.

    Notes
    -----
    - This function uses the initial conditions y0 = [0, 0] for the numerical
      solver. These correspond to a gravitational potential and its derivative
      of zero at the initial radius.

    - The initial radius and maximum radius for the integration are hard-coded 
      into the function as r_init = 1e-2 pc and r_max = 1e8 pc, respectively.

    - If r contains zero(s), this function sets the zero(s) to r_init to avoid 
      division by zero.
    """

    r_init = 1e-2  # pc
    r_max = 1e8  # pc

    # Avoid division by zero
    if np.min(r) == 0:
        r[np.argmin(r)] = r_init

    # ODE Initial conditions
    y0 = [0, 0]

    # Use ODE solver to solve Poisson's equation
    y_num_infty = odeint(solve_poisson, y0, [r_init, r_max],
                         args=(rho_func, *rho_params))
    y_numerical = odeint(solve_poisson, y0, np.append(r_init, r),
                         args=(rho_func, *rho_params))
    phi_numerical = y_numerical[1:, 0] - y_num_infty[-1, 0]

    return phi_numerical
