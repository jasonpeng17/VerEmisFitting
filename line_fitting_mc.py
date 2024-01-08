# The code provided defines a class called line_fitting_mc, which contains methods for calculating the Monte Carlo error of various line ratios, ne, Te, etc.

import numpy as np
import pyneb as pn
from astropy.io import fits
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy import integrate
import copy
import lmfit
from lmfit import (minimize, Parameters, Minimizer, conf_interval, conf_interval2d,
                   report_ci, report_fit, fit_report)
from astropy import constants as const
from scipy.stats import f
from astropy.stats import sigma_clip, sigma_clipped_stats

class line_fitting_mc():
    def __init__(self, seed = 42):
        # Description
        # This is a constructor method that initializes the class variables self.rng. Here self.rng is a random number generator object.
        self.rng = np.random.default_rng(seed) 

    # this method is used to calculate the error of local continuum, but might overestimate the error (deprecated)
    def flux_cont_MC(self, n, wave, p0, p1, std_p0, std_p1):
        '''
        This function calculates the uncertainties in a linear continuum fit to a spectrum by running Monte Carlo simulations. It takes in the following parameters:
        
        Parameters
        ----------
        self: the object instance of the class this method is defined in.
        n: the number of Monte Carlo simulations to run.
        wave: the wavelength array of the spectrum.
        p0 and p1: the linear fit parameters (y = p1 * x + p0) of the continuum.
        std_p0 and std_p1: the uncertainties in the linear fit parameters.

        This function generates random values for p0 and p1 based on their uncertainties and calculates the linear continuum fit for each Monte Carlo simulation. 
        It returns an array of the mean uncertainties in the continuum fit for each wavelength value.
        '''
        n_monte_carlo = n
        self.result_flux_cont = np.zeros((len(wave), n_monte_carlo))
        cont_err_lower = np.zeros(len(wave))
        cont_err_upper = np.zeros(len(wave))
        p0_lower = p0 - std_p0
        p0_upper = p0 + std_p0
        p1_lower = p1 - std_p1
        p1_upper = p1 + std_p1
        
        for i in range(n_monte_carlo):
            p0_mc = p0_lower + np.float64(self.rng.random(1)*(p0_upper-p0_lower))
            p1_mc = p1_lower + np.float64(self.rng.random(1)*(p1_upper-p1_lower))
            self.result_flux_cont[:, i] = np.polyval(np.array([p1_mc, p0_mc]), wave)
        for ii in range(len(wave)):
            cont_err_lower[ii] = np.nanpercentile(self.result_flux_cont[ii, :],50) - np.nanpercentile(self.result_flux_cont[ii, :],16)
            cont_err_upper[ii] = np.nanpercentile(self.result_flux_cont[ii, :],84) - np.nanpercentile(self.result_flux_cont[ii, :],50)
        cont_err_mean = (cont_err_lower + cont_err_upper) / 2.
        return cont_err_mean

    # this method is used to calculate the error of line flux integrated from scipy.integrate.quad for the Lorentzian model that do not have analytical solution
    def line_flux_lorentz_MC(self, n, x0, sigma, amp, x0_err, sigma_err, amp_err):
        '''
        This function calculates the uncertainties line flux integrated from scipy.integrate.quad for the Lorentzian model that do not have analytical solution:
        
        Parameters
        ----------
        self: the object instance of the class this method is defined in.
        n: the number of Monte Carlo simulations to run.
        x0 and x0_err: the center of the Lorentzian profile and its error.
        sigma and sigma_err: the width of the Lorentzian profile and its error (FWHM = 2 * sigma).
        amp and amp_err: the amplitude of the Lorentzian profile and its error.
        '''
        x0_lower = x0 - x0_err
        x0_upper = x0 + x0_err
        sigma_lower = sigma - sigma_err
        sigma_upper = sigma + sigma_err
        amp_lower = amp - amp_err
        amp_upper = amp + amp_err
        self.result_line_flux = np.zeros(n)

        for i in range(n):
            x0_mc = x0_lower + np.float64(self.rng.random(1)*(x0_upper-x0_lower))
            sigma_mc = sigma_lower + np.float64(self.rng.random(1)*(sigma_upper-sigma_lower))
            amp_mc = amp_lower + np.float64(self.rng.random(1)*(amp_upper-amp_lower))
            self.result_line_flux[i] = integrate.quad(lorentzian_1p_v, -np.inf, np.inf, args=(x0_mc, sigma_mc, amp_mc))[0]
        line_flux_lerr = np.nanpercentile(self.result_line_flux,50) - np.nanpercentile(self.result_line_flux,16)
        line_flux_uerr = np.nanpercentile(self.result_line_flux,84) - np.nanpercentile(self.result_line_flux,50)
        line_flux_err_ave = (line_flux_lerr + line_flux_uerr) / 2.
        return np.array([line_flux_lerr, line_flux_uerr]), line_flux_err_ave

