# The code provided defines a class called line_fitting_mc, which contains methods for calculating the Monte Carlo error of various line ratios, ne, Te, etc.

import numpy as np
import pyneb as pn
from astropy.io import fits
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import copy
import lmfit
from lmfit import (minimize, Parameters, Minimizer, conf_interval, conf_interval2d,
                   report_ci, report_fit, fit_report)
from scipy import integrate
from astropy import constants as const
from scipy.stats import f
from astropy.stats import sigma_clip, sigma_clipped_stats

from IPython import embed

class line_fitting_mc():
    def __init__(self, seed = 42):
        # Description
        # This is a constructor method that initializes the class variables self.rng. Here self.rng is a random number generator object.
        self.rng = np.random.default_rng(seed) 

    def Hb_Hy_err_MC(self, n, flux_Hb, flux_Hy, Hb_flux_err, Hy_flux_err):
        # Description
        # This method takes in the number of Monte Carlo iterations n, and the measured fluxes and their respective errors for the Hydrogen beta (Hβ) 
        # and Hydrogen gamma (Hγ) lines: flux_Hb, flux_Hy, Hb_flux_err, and Hy_flux_err. It then performs Monte Carlo simulations by randomly sampling 
        # flux values from Gaussian distributions around flux_Hb and flux_Hy with standard deviations of Hb_flux_err and Hy_flux_err, respectively. 
        # The method calculates the ratio of Hβ to Hγ fluxes for each simulation and returns the standard deviation of the resulting distribution of flux ratios.
        n_monte_carlo = n
        Hb_flux_lower = flux_Hb - Hb_flux_err
        Hb_flux_upper = flux_Hb + Hb_flux_err
        Hy_flux_lower = flux_Hy - Hy_flux_err
        Hy_flux_upper = flux_Hy + Hy_flux_err
        self.result_ratio_HbHy = np.ones(n_monte_carlo)
        for i in range(n_monte_carlo):
            flux_mc_Hb = Hb_flux_lower + np.float64(self.rng.random(1)*(Hb_flux_upper-Hb_flux_lower))
            flux_mc_Hy = Hy_flux_lower + np.float64(self.rng.random(1)*(Hy_flux_upper-Hy_flux_lower))
            self.result_ratio_HbHy[i] = flux_mc_Hb/flux_mc_Hy
        HbHy_ratio_err_lower = np.percentile(self.result_ratio_HbHy,50) - np.percentile(self.result_ratio_HbHy,16)
        HbHy_ratio_err_upper = np.percentile(self.result_ratio_HbHy,84) - np.percentile(self.result_ratio_HbHy,50)
        HbHy_ratio_std = np.array([HbHy_ratio_err_lower,HbHy_ratio_err_upper])
        return HbHy_ratio_std
    
    def Hy_Hd_err_MC(self, n, flux_Hy, flux_Hd, Hy_flux_err, Hd_flux_err):
        # Description 
        # This method takes in the number of Monte Carlo iterations n, and the measured fluxes and their respective errors for the Hydrogen gamma (Hγ) 
        # and Hydrogen delta (Hδ) lines: flux_Hy, flux_Hd, Hy_flux_err, and Hd_flux_err. It then performs Monte Carlo simulations by randomly sampling 
        # flux values from Gaussian distributions around flux_Hy and flux_Hd with standard deviations of Hy_flux_err and Hd_flux_err, respectively. 
        # The method calculates the ratio of Hγ to Hδ fluxes for each simulation and returns the standard deviation of the resulting distribution of flux ratios.
        n_monte_carlo = n
        Hd_flux_lower = flux_Hd - Hd_flux_err
        Hd_flux_upper = flux_Hd + Hd_flux_err
        Hy_flux_lower = flux_Hy - Hy_flux_err
        Hy_flux_upper = flux_Hy + Hy_flux_err
        self.result_ratio_HyHd = np.ones(n_monte_carlo)
        
        for i in range(n_monte_carlo):
            flux_mc_Hd = Hd_flux_lower + np.float64(self.rng.random(1)*(Hd_flux_upper-Hd_flux_lower))
            flux_mc_Hy = Hy_flux_lower + np.float64(self.rng.random(1)*(Hy_flux_upper-Hy_flux_lower))
            self.result_ratio_HyHd[i] = flux_mc_Hy/flux_mc_Hd
        HyHd_ratio_err_lower = np.percentile(self.result_ratio_HyHd,50) - np.percentile(self.result_ratio_HyHd,16)
        HyHd_ratio_err_upper = np.percentile(self.result_ratio_HyHd,84) - np.percentile(self.result_ratio_HyHd,50)
        HyHd_ratio_std = np.array([HyHd_ratio_err_lower,HyHd_ratio_err_upper])
        return HyHd_ratio_std
    
    def Hb_Hy_Hd_err_MC(self, n, flux_Hb, flux_Hy, flux_Hd, Hb_flux_err, Hy_flux_err, Hd_flux_err):
        # Description
        # combination of the Hb_Hy_err_MC and Hy_Hd_err_MC functions. 

        n_monte_carlo = n
        
        Hb_flux_lower = flux_Hb - Hb_flux_err
        Hb_flux_upper = flux_Hb + Hb_flux_err
        Hd_flux_lower = flux_Hy - Hd_flux_err
        Hd_flux_upper = flux_Hy + Hd_flux_err
        Hy_flux_lower = flux_Hd - Hy_flux_err
        Hy_flux_upper = flux_Hd + Hy_flux_err
        self.result_ratio_HbHy = np.ones(n_monte_carlo)
        self.result_ratio_HyHd = np.ones(n_monte_carlo)
        
        for i in range(n_monte_carlo):
            flux_mc_Hb = Hb_flux_lower + np.float64(self.rng.random(1)*(Hb_flux_upper-Hb_flux_lower))
            flux_mc_Hd = Hd_flux_lower + np.float64(self.rng.random(1)*(Hd_flux_upper-Hd_flux_lower))
            flux_mc_Hy = Hy_flux_lower + np.float64(self.rng.random(1)*(Hy_flux_upper-Hy_flux_lower))
            self.result_ratio_HbHy[i] = flux_mc_Hb/flux_mc_Hy
            self.result_ratio_HyHd[i] = flux_mc_Hy/flux_mc_Hd
        HbHy_ratio_err_lower = np.percentile(self.result_ratio_HbHy,50) - np.percentile(self.result_ratio_HbHy,16)
        HbHy_ratio_err_upper = np.percentile(self.result_ratio_HbHy,84) - np.percentile(self.result_ratio_HbHy,50)
        HyHd_ratio_err_lower = np.percentile(self.result_ratio_HyHd,50) - np.percentile(self.result_ratio_HyHd,16)
        HyHd_ratio_err_upper = np.percentile(self.result_ratio_HyHd,84) - np.percentile(self.result_ratio_HyHd,50)
        HbHy_ratio_std = np.array([HbHy_ratio_err_lower,HbHy_ratio_err_upper])
        HyHd_ratio_std = np.array([HyHd_ratio_err_lower,HyHd_ratio_err_upper])
        return HbHy_ratio_std, HyHd_ratio_std

    def O2_low_Hbeta_MC(self,n, flux_3726, flux_3729, Hb_flux, flux_3726_err, flux_3729_err, Hb_flux_err): 
        # Description
        # This function calculates the ratio between the [OII] 3726 and 3729 emission lines and the H-beta (Hβ) emission line. 
        # The function takes as input the number of Monte Carlo simulations to perform (n), the fluxes and flux errors for the [OII] and Hβ lines, and 
        # returns the error in the [OII]/Hβ ratio.
        n_monte_carlo_O2 = n
        self.result_ratio_O2Hb = np.ones(n_monte_carlo_O2) 
        flux_3726_lower = flux_3726 - flux_3726_err
        flux_3726_upper = flux_3726 + flux_3726_err
        flux_3729_lower = flux_3729 - flux_3729_err
        flux_3729_upper = flux_3729 + flux_3729_err
        Hb_flux_lower = Hb_flux - Hb_flux_err
        Hb_flux_upper = Hb_flux + Hb_flux_err
        
        for i in range(n_monte_carlo_O2):
            flux_mc_3726 = flux_3726_lower + np.float64(self.rng.random(1)*(flux_3726_upper-flux_3726_lower))
            flux_mc_3729 = flux_3729_lower + np.float64(self.rng.random(1)*(flux_3729_upper-flux_3729_lower))
            flux_mc_Hb = Hb_flux_lower + np.float64(self.rng.random(1)*(Hb_flux_upper-Hb_flux_lower))
            self.result_ratio_O2Hb[i] = (flux_mc_3726+flux_mc_3729)/(flux_mc_Hb)
        for i in range(len(self.result_ratio_O2Hb)):
            if self.result_ratio_O2Hb[i] < 0:
                self.result_ratio_O2Hb[i] = 0
        self.result_ratio_O2Hb = np.ma.masked_values(self.result_ratio_O2Hb,0,shrink=False).compressed()
        O2Hb_ratio_err_lower = np.nanpercentile(self.result_ratio_O2Hb,50) - np.nanpercentile(self.result_ratio_O2Hb,16)
        O2Hb_ratio_err_upper = np.nanpercentile(self.result_ratio_O2Hb,84) - np.nanpercentile(self.result_ratio_O2Hb,50)
        O2Hb_ratio_std = np.array([O2Hb_ratio_err_lower,O2Hb_ratio_err_upper])
        return O2Hb_ratio_std
    
    def O3_Hbeta_MC(self,n, flux_4959, flux_5007, Hb_flux, flux_4959_err, flux_5007_err, Hb_flux_err):
        # Description
        # This function calculates the ratio between the [OIII] 4959 and 5007 emission lines and the Hβ emission line. 
        # The function takes as input the number of Monte Carlo simulations to perform (n), the fluxes and flux errors for the [OIII] and Hβ lines, 
        # and returns the error in the [OIII]/Hβ ratio.
        n_monte_carlo_O3 = n
        self.result_ratio_O3Hb = np.ones(n_monte_carlo_O3)
        flux_4959_lower = flux_4959 - flux_4959_err
        flux_4959_upper = flux_4959 + flux_4959_err
        flux_5007_lower = flux_5007 - flux_5007_err
        flux_5007_upper = flux_5007 + flux_5007_err
        Hb_flux_lower = Hb_flux - Hb_flux_err
        Hb_flux_upper = Hb_flux + Hb_flux_err
        
        for i in range(n_monte_carlo_O3):
            flux_mc_4959 = flux_4959_lower + np.float64(self.rng.random(1)*(flux_4959_upper-flux_4959_lower))
            flux_mc_5007 = flux_5007_lower + np.float64(self.rng.random(1)*(flux_5007_upper-flux_5007_lower))
            flux_mc_Hb = Hb_flux_lower + np.float64(self.rng.random(1)*(Hb_flux_upper-Hb_flux_lower))
            self.result_ratio_O3Hb[i] = (flux_mc_4959+flux_mc_5007)/flux_mc_Hb
        O3Hb_ratio_err_lower = np.nanpercentile(self.result_ratio_O3Hb,50) - np.nanpercentile(self.result_ratio_O3Hb,16)
        O3Hb_ratio_err_upper = np.nanpercentile(self.result_ratio_O3Hb,84) - np.nanpercentile(self.result_ratio_O3Hb,50)
        O3Hb_ratio_std = np.array([O3Hb_ratio_err_lower,O3Hb_ratio_err_upper])
        return O3Hb_ratio_std

    def O2_ne_MC(self, n, flux_3726, flux_3729, flux_3726_err, flux_3729_err, tem = 20000.):
        # Description 
        '''
        Perform a Monte Carlo simulation to estimate the OII nebular electron density and its error.
        
        Parameters
        ----------
        n : int
            Number of Monte Carlo simulations to perform.
        flux_3726 : float
            Flux of the [OII]3726 line.
        flux_3729 : float
            Flux of the [OII]3729 line.
        flux_3726_err : float
            Error on the flux of the [OII]3726 line.
        flux_3729_err : float
            Error on the flux of the [OII]3729 line.
        tem : float, optional
            Electron temperature of the nebula, in K. Default is 20000.
            
        Returns
        -------
        np.array([O2_ne_std,O2_ratio_std]) : numpy array
            The estimated OII electron density and its error, in cm^-3.
        '''
        O2 = pn.Atom('O',2)
        n_monte_carlo_O2 = n
        self.result_ratio_O2 = np.ones(n_monte_carlo_O2)
        self.result_ne_O2 = np.ones(n_monte_carlo_O2)
        flux_3726_lower = flux_3726 - flux_3726_err
        flux_3726_upper = flux_3726 + flux_3726_err
        flux_3729_lower = flux_3729 - flux_3729_err
        flux_3729_upper = flux_3729 + flux_3729_err
     
        for i in range(n_monte_carlo_O2):        
            flux_mc_3726 = flux_3726_lower + np.float64(self.rng.random(1)*(flux_3726_upper-flux_3726_lower))
            flux_mc_3729 = flux_3729_lower + np.float64(self.rng.random(1)*(flux_3729_upper-flux_3729_lower))
            self.result_ratio_O2[i] = flux_mc_3729/flux_mc_3726
            self.result_ne_O2[i] = O2.getTemDen(self.result_ratio_O2[i], tem = tem,to_eval="I(2, 1) / I(3, 1)" )
        O2_ratio_err_lower = np.nanpercentile(self.result_ratio_O2,50) - np.nanpercentile(self.result_ratio_O2,16)
        O2_ratio_err_upper = np.nanpercentile(self.result_ratio_O2,84) - np.nanpercentile(self.result_ratio_O2,50)
        O2_ratio_std = np.array([O2_ratio_err_lower,O2_ratio_err_upper])
        if len(self.result_ne_O2) != 0:
            O2_ne_err_lower = np.nanpercentile(self.result_ne_O2,50) - np.nanpercentile(self.result_ne_O2,16)
            O2_ne_err_upper = np.nanpercentile(self.result_ne_O2,84) - np.nanpercentile(self.result_ne_O2,50)
            O2_ne_std = np.array([O2_ne_err_lower,O2_ne_err_upper])
        elif len(self.result_ne_O2) == 0:
            O2_ne_std = np.array([np.nan,np.nan])
        return np.array([O2_ne_std,O2_ratio_std])


    def O3_Te_MC(self, n, den, flux_4363, flux_4959, flux_5007, flux_4363_err, flux_4959_err, flux_5007_err):
        # Description 
        '''
        This function calculates the statistical error of electron temperature using the [O III] forbidden line ratio method. It takes in the following parameters:
        
        Parameters
        ----------
        self: the object instance of the class this method is defined in.
        n: the number of Monte Carlo simulations to run.
        den: the electron density of the gas.
        flux_4363, flux_4959, and flux_5007: the observed fluxes of the [O III] emission lines at 4363 Å, 4959 Å, and 5007 Å, respectively.
        flux_4363_err, flux_4959_err, and flux_5007_err: the uncertainties in the observed fluxes of the [O III] emission lines at 4363 Å, 4959 Å, and 5007 Å, respectively.
        
        This function generates random flux values for each of the three emission lines, calculates the [O III] line ratio, and then uses the 
        PyNeb package to calculate the electron temperature using the line ratio and the given electron density. 
        It returns an array of two arrays, where the first array contains the error bars for the calculated electron temperature and 
        the second array contains the error bars for the [O III] line ratio.
        '''
        O3 = pn.Atom('O',3)
        n_monte_carlo_O3 = n
        self.result_ratio_O3 = np.ones(n_monte_carlo_O3)
        self.result_Te_O3 = np.ones(n_monte_carlo_O3)
        flux_4363_lower = flux_4363 - flux_4363_err
        flux_4363_upper = flux_4363 + flux_4363_err
        flux_4959_lower = flux_4959 - flux_4959_err
        flux_4959_upper = flux_4959 + flux_4959_err
        flux_5007_lower = flux_5007 - flux_5007_err
        flux_5007_upper = flux_5007 + flux_5007_err
        
        for i in range(n_monte_carlo_O3):
            flux_mc_4363 = flux_4363_lower + np.float64(self.rng.random(1)*(flux_4363_upper-flux_4363_lower))
            flux_mc_4959 = flux_4959_lower + np.float64(self.rng.random(1)*(flux_4959_upper-flux_4959_lower))
            flux_mc_5007 = flux_5007_lower + np.float64(self.rng.random(1)*(flux_5007_upper-flux_5007_lower))
            self.result_ratio_O3[i] = (flux_mc_4959+flux_mc_5007)/flux_mc_4363
            self.result_Te_O3[i] = O3.getTemDen(self.result_ratio_O3[i], den = den,to_eval="(I(4, 3) + I(4, 2))/ I(5, 4)" )
        O3_ratio_err_lower = np.nanpercentile(self.result_ratio_O3,50) - np.nanpercentile(self.result_ratio_O3,16)
        O3_ratio_err_upper = np.nanpercentile(self.result_ratio_O3,84) - np.nanpercentile(self.result_ratio_O3,50)
        O3_ratio_std = np.array([O3_ratio_err_lower,O3_ratio_err_upper])
        O3_Te_err_lower = np.nanpercentile(self.result_Te_O3,50) - np.nanpercentile(self.result_Te_O3,16)
        O3_Te_err_upper = np.nanpercentile(self.result_Te_O3,84) - np.nanpercentile(self.result_Te_O3,50)
        O3_Te_std = np.array([O3_Te_err_lower,O3_Te_err_upper])
        return np.array([O3_Te_std,O3_ratio_std])

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







