#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:30:36 2023

@author: kacharov
"""

import numpy as np
from scipy.integrate import quad
from astropy import units as u

# Constants
G = 4.3009172706e-3  # Gravitational constant, pc Msun^-1 (km/s)^2


def gnfw_density(r, rho_s, r_s, gamma):
    """
    Calculate the NFW density at a given radius.

    Parameters:
    r : float
        The radius at which to calculate the density (Mpc).
    rho_s : float
        The scale density (solar masses / Mpc^3).
    r_s : float
        The scale radius (Mpc).
    gamma : float
        The central density slope.

    Returns:
    float
        The NFW density at the given radius (solar masses / Mpc^3).
    """
    return rho_s / ((r/r_s)**gamma * (1 + r/r_s)**(3-gamma))


def gnfw_mass(r, rho_s, r_s, gamma):
    """
    Calculate the cumulative mass within a given radius for a NFW profile.

    Parameters:
    r : float
        The radius within which to calculate the mass (Mpc).
    rho_s : float
        The scale density (solar masses / Mpc^3).
    r_s : float
        The scale radius (Mpc).
    gamma : float
        The central density slope.

    Returns:
    float
        The cumulative mass within the given radius (solar masses).
    """

    if gamma == 1:
        integral = np.log(1 + r/r_s) - r/(r_s + r)
    elif gamma == 0:
        integral = -(r*(2*r_s + 3*r))/(2*(r_s + r)**2) + \
            np.log(r_s + r) - np.log(r_s)
    else:
        print("I provide only a numerical solution for gamma!=0 and gamma!=1")
        def func(x): return x**2 * (x**(-gamma) * (1 + x)**(gamma-3))
        integral = np.zeros(len(r))
        for i in range(0, len(r)):
            integral[i] = quad(func, 0, r[i]/r_s)[0]

    return 4 * np.pi * rho_s * r_s**3 * integral


def gnfw_potential(r, rho_s, r_s, gamma):
    """
    Calculates the gravitational potential for the generalized
    Navarro-Frenk-White (gNFW) profile.

    The gNFW profile gives the density as a function of radius.
    This function calculates the potential for 
    two particular cases of the gNFW profile,
    specified by the gamma parameter: gamma = 1 and gamma = 0.

    Parameters
    ----------
    r : array_like
        Array of radial distances at which to compute the potential.

    rho_s : float
        Scale density parameter of the gNFW profile.

    r_s : float
        Scale radius parameter of the gNFW profile.

    gamma : int, {0,1}
        Index of the gNFW profile. This function only provides analytical
        solutions for gamma = 0 and gamma = 1.

    Returns
    -------
    potential : numpy.ndarray
        The gravitational potential corresponding to the particular
        gNFW profile, computed at each radius in r.

    Notes
    -----
    - If r contains zero(s), this function sets the zero(s) to a very small
    number (1e-10) to avoid division by zero.

    - If gamma is neither 0 nor 1, this function prints an error message and
    does not return a potential.
    """

    # Avoid division by zero
    if np.min(r) == 0:
        r[np.argmin(r)] = 1e-10

    if gamma == 1:
        return -4 * np.pi * G * rho_s * r_s**3 * np.log(1 + r / r_s) / r
    elif gamma == 0:
        return -4 * np.pi * G * rho_s * r_s**2 * (r_s/r * np.log(1+r/r_s) -
                                                  0.5/(1+r/r_s))
    else:
        print("I provide only the analytical solutions for gamma==0 and gamma==1")


def gnfw_params(M200, concentration, gamma=1, H0=70*u.km/u.s/u.Mpc):
    """
    Calculate the central (scale) density and scale radius of a NFW halo.
    Refrence Wikipedia:
    https://en.wikipedia.org/wiki/Navarro–Frenk–White_profile

    Parameters:
    M200 : quantity
         The characteristic mass of the halo (solar masses).
    concentration : float
         The concentration of the halo.
    gamma : float
        The central density slope.
    H : float, optional
         The Hubble parameter (km s^-1 Mpc^-1). Default is H0=70.

    Returns:
    rho_s : float
         The central density of the halo (solar masses / Mpc^3).
    r_s : float
         The scale radius of the halo (Mpc).
    """

    # The gravitational constant (km^2 Mpc/(s^2 solar masses))
    G = 4.301e-9 * u.km**2 * u.Mpc / u.s**2 / u.Msun

    # Convert Hubble parameter to s^-1
    H0 = H0.to("1/s")  # km/s/Mpc to s^-1

    # Calculate the critical density
    rho_crit = 3 * (H0**2) / (8 * np.pi * G)

    # Calculate the characteristic overdensity (assume Δ_c=200)
    delta_c = 200

    # Calculate R200
    R200 = ((3 * M200) / (4 * np.pi * delta_c * rho_crit))**(1/3)
    R200 = R200.to("pc")

    # Calculate the scale radius
    r_s = R200 / concentration

    # Calculate the central density
    if gamma == 1:
        integral = np.log(1 + concentration) - \
            concentration / (1 + concentration)
    elif gamma == 0:
        integral = np.log(1 + concentration) - \
            0.5*concentration * (2 + 3*concentration)/(1 + concentration)**2
    else:
        integral = quad(lambda x: x**2 * (x**(-gamma) * (1 + x)**(gamma-3)),
                        0, concentration)[0]

    rho_s = M200 / (4 * np.pi * r_s**3 * integral)

    return rho_s, r_s
