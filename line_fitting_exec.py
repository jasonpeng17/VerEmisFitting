import numpy as np
from astropy.io import fits
import os, sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import skew
import copy
import lmfit
from lmfit import (minimize, Parameters, Minimizer, conf_interval, conf_interval2d,
                   report_ci, report_fit, fit_report)
from scipy import integrate
from astropy import constants as const
from scipy.stats import f
from astropy.stats import sigma_clip, sigma_clipped_stats
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
import tkinter as tk
from tkinter.simpledialog import askstring

from line_fitting_model import *
from line_fitting_mc import *
from line_fitting_prelim import *
from Interstellar_Extinction import *
from select_gui import *

from IPython import embed

class line_fitting_exec():
    def __init__(self, redshift = None, E_BV = None, A_V = None, vac_or_air = 'air', seed = 42, fits_name = None, line_select_method = 'gui', input_txt = None):
        """
        Constructor for the line_fitting_exec class that initializes the following class variables:
        
        c : float
            The speed of light in km/s.
        rng : numpy.random.Generator
            The random number generator.
        E_BV : float
            The MW reddening of the galaxy.
        A_V : float
            The intrinsic extinction of the galaxy.
        redshift : float
            The redshift of the galaxy.
        """
        self.c = const.c.to('km/s').value
        # save the seed value and apply to the line_fitting_model function such that it has the same randomness
        self.seed = seed
        self.rng = np.random.default_rng(self.seed) 
        # define the MW and intrinsic extinction of the galaxy
        self.E_BV = E_BV
        self.A_V = A_V
        if redshift == None:
            self.redshift = 0. # assume it is already rest-frame
        if redshift != None:
            self.redshift = redshift

        # get the rest-frame and observed-frame wavelength of each intended line
        # check whether the observed frame is in vacuum 'vac' or in 'air' (ADD any intended lines for fittings)
        if vac_or_air == 'air':
            # air wavelengths
            self.wave_dict = {
                'H delta': np.array([4101.76, 4101.76*(1+self.redshift)]),
                'H gamma': np.array([4340.47, 4340.47*(1+self.redshift)]),
                'H beta': np.array([4861.33, 4861.33*(1+self.redshift)]),
                'H alpha': np.array([6562.80, 6562.80*(1+self.redshift)]),
                'H nu': np.array([3703.86, 3703.86*(1+self.redshift)]),
                'H kappa': np.array([3750.15, 3750.15*(1+self.redshift)]),
                'H iota': np.array([3770.63, 3770.63*(1+self.redshift)]),
                '[OI] 6300': np.array([6300.30, 6300.30*(1+self.redshift)]),
                '[OII]&[OII] 3726&3729': np.array([3726.03, 3728.82, 3726.03*(1+self.redshift), 3728.82*(1+self.redshift)]),
                '[OIII]&HeI 5007&5015': np.array([5006.84, 5015.6776, 5006.84*(1+self.redshift), 5015.6776*(1+self.redshift)]),
                '[NII]&H&[NII] 6548&alpha&6583' : np.array([6548.03, 6562.80, 6583.41, 6548.03*(1+self.redshift), 6562.80*(1+self.redshift), 6583.41*(1+self.redshift)]), 
                '[SII]&[SII] 6716&6731': np.array([6716.47, 6730.85, 6716.47*(1+self.redshift), 6730.85*(1+self.redshift)]),
                '[OIII] 4363': np.array([4363.21, 4363.21*(1+self.redshift)]),
                '[OIII] 4959': np.array([4958.92, 4958.92*(1+self.redshift)]),
                '[OIII] 5007': np.array([5006.84, 5006.84*(1+self.redshift)]),
                'HeII 4686': np.array([4685.68, 4685.68*(1+self.redshift)]),
                'HeI 5015': np.array([5015.6776, 5015.6776*(1+self.redshift)]),
                '[ArIV] 4711': np.array([4711.35, 4711.35*(1+self.redshift)]),
                '[ArIV] 4740': np.array([4740.20, 4740.20*(1+self.redshift)]),
                '[ArIV]&HeI 4711&4713': np.array([4711.35, 4713.1392, 4711.35*(1+self.redshift), 4713.1392*(1+self.redshift)])
            }
        if vac_or_air == 'vac':
            # vacuum wavelengths
            self.wave_dict = {
                'H delta': np.array([4102.92, 4102.92*(1+self.redshift)]),
                'H gamma': np.array([4341.69, 4341.69*(1+self.redshift)]),
                'H beta': np.array([4862.69, 4862.69*(1+self.redshift)]),
                'H alpha': np.array([6564.61, 6564.61*(1+self.redshift)]),
                'H nu': np.array([3704.90, 3704.90*(1+self.redshift)]),
                'H kappa': np.array([3751.22, 3751.22*(1+self.redshift)]),
                'H iota': np.array([3771.70, 3771.70*(1+self.redshift)]),
                '[OI] 6300': np.array([6302.04, 6302.04*(1+self.redshift)]),
                '[OII]&[OII] 3726&3729': np.array([3727.09, 3729.88, 3727.09*(1+self.redshift), 3729.88*(1+self.redshift)]),
                '[OIII]&HeI 5007&5015': np.array([5008.24, 5017.0767, 5008.24*(1+self.redshift), 5017.0767*(1+self.redshift)]),
                '[NII]&H&[NII] 6548&alpha&6583' : np.array([6549.84, 6564.61, 6585.23, 6549.84*(1+self.redshift), 6564.61*(1+self.redshift), 6585.23*(1+self.redshift)]), 
                '[SII]&[SII] 6716&6731': np.array([6718.32, 6732.71, 6718.32*(1+self.redshift), 6732.71*(1+self.redshift)]),
                '[OIII] 4363': np.array([4364.44, 4364.44*(1+self.redshift)]),
                '[OIII] 4959': np.array([4960.30, 4960.30*(1+self.redshift)]),
                '[OIII] 5007': np.array([5008.24, 5008.24*(1+self.redshift)]),
                'HeII 4686': np.array([4686.99, 4686.99*(1+self.redshift)]),
                'HeI 5015': np.array([5017.0767, 5017.0767*(1+self.redshift)]),
                '[ArIV] 4711': np.array([4712.67, 4712.67*(1+self.redshift)]),
                '[ArIV] 4740': np.array([4741.53, 4741.53*(1+self.redshift)]),
                '[ArIV]&HeI 4711&4713': np.array([4712.67, 4714.4580, 4712.67*(1+self.redshift), 4714.4580*(1+self.redshift)])
            }

        # emission lines available for fitting
        self.emission_lines_lst = list(self.wave_dict.keys())

        # Mapping from line name to method
        # ADD any intended lines for fittings
        self.lines_dict = {
            '[OII]&[OII] 3726&3729': self.O2_doublet_low_return, '[OIII] 4363': self.O3_4363_return, '[OI] 6300': self.O1_6300_return,
            '[OIII]&HeI 5007&5015': self.O3_5007_HeI_5015_return, '[NII]&H&[NII] 6548&alpha&6583': self.N2_6548_Ha_N2_6583_return,
            '[OIII] 4959': self.O3_4959_return, '[OIII] 5007': self.O3_5007_return, '[SII]&[SII] 6716&6731': self.S2_doublet_return,
            'H nu': self.Hv_return, 'H kappa': self.Hk_return, 'H iota': self.Hi_return,
            'H delta': self.Hd_return, 'H gamma': self.Hy_return, 'H beta': self.Hb_return, 'H alpha': self.Ha_return,
            'HeI 5015': self.He1_5015_return, 'HeII 4686': self.He2_4686_return, '[ArIV] 4711': self.Ar4_4711_return, '[ArIV] 4740': self.Ar4_4740_return,
            '[ArIV]&HeI 4711&4713': self.Ar4_4711_HeI_4713_return
        }

        # initialize the line-continuum dict for all intended lines
        self.cont_line_dict = dict()

        # initialize a gui for users to select the lines to be fitted
        # it also allows users to choose whether to fix the velocity centroid or width during fittings
        if line_select_method == 'gui':
            self.selector = LineSelector(self.emission_lines_lst)
            self.selected_lines, self.fitting_method, self.broad_wings_lines, self.double_gauss_lines, self.triple_gauss_lines, self.absorption_lines = \
            self.selector.run() # selected lines and the selected fitting method

            # whether the second or the third emission components are broad or not
            self.double_gauss_broad = True if self.selector.double_gauss_broad == 'yes' else False 
            self.triple_gauss_broad = True if self.selector.triple_gauss_broad == 'yes' else False 

            # get the initial guess, range size for each parameter, and also the fixed ratios for selected amplitude pairs
            self.initial_guess_dict, self.param_range_dict, self.amps_ratio_dict = self.selector.params_dict, self.selector.params_range_dict, self.selector.amps_ratio_dict

        # users need to input a line selection text for this approach
        if line_select_method == 'txt':
            line_select_pars = extract_line_pars(input_txt)
            # selected lines and the selected fitting method
            self.selected_lines = line_select_pars['selected_lines']
            self.fitting_method = line_select_pars['fitting_method']
            self.broad_wings_lines = line_select_pars['multi_emis_lines']
            self.double_gauss_lines = line_select_pars['double_gauss_lines']
            self.triple_gauss_lines = line_select_pars['triple_gauss_lines']
            self.absorption_lines = line_select_pars['absorption_lines']
            # whether the second or the third emission components are broad or not
            self.double_gauss_broad = line_select_pars['double_gauss_broad']
            self.triple_gauss_broad = line_select_pars['triple_gauss_broad']

            # Call FitParamsWindow
            fit_window = FitParamsWindow(self.selected_lines, self.broad_wings_lines, self.triple_gauss_lines, self.absorption_lines)
            # get the initial guess, range size for each parameter, and also the fixed ratios for selected amplitude pairs
            self.initial_guess_dict, self.param_range_dict, self.amps_ratio_dict = fit_window.run()

        # get the name of the input fits file (excluding .fits)
        self.fits_name = fits_name.split('/')[-1][:-5] if fits_name != None else None

    ###### beginning of functions from line_fitting
    def find_two_peaks(self, list1):
        """
        Given a list, returns the two highest values in the list.
        """
        mx=max(list1[0],list1[1])  
        secondmax=min(list1[0],list1[1])  
        n =len(list1) 
        for i in range(2,n):  
            if list1[i]>mx:  
                secondmax=mx 
                mx=list1[i]  
            elif list1[i]>secondmax and mx != list1[i]:  
                secondmax=list1[i] 
            else: 
                if secondmax == mx: 
                    secondmax = list1[i]
        result = [mx,secondmax]
        return result

    def max_spacing(self, wave):
        """
        Given a wavelength array, returns the maximum spacing between two consecutive elements.
        """
        space_array = np.diff(wave)
        max_spacing = np.max(space_array)
        return max_spacing
    
    def region_around_line(self, w, flux, cont, order = 2, sigma_clip_or_not = True):
        #index is true in the region where we fit the polynomial
        indcont = ((w >= cont[0][0]) & (w <= cont[0][1])) |((w >= cont[1][0]) & (w <= cont[1][1]))
        #index of the region we want to return
        indrange = (w >= cont[0][0]) & (w <= cont[1][1])
        # make a flux array of shape
        # (number of spectra, number of points in indrange)
        f = np.zeros(indrange.sum())
        if sigma_clip_or_not:
            filtered_flux = sigma_clip(flux[indcont], sigma=3, maxiters=None, cenfunc='median', masked=True, copy=False)
            filtered_wave = np.ma.compressed(np.ma.masked_array(w[indcont], filtered_flux.mask))
            filtered_flux_data = filtered_flux.compressed()
            # fit polynomial of second order to the continuum region
            linecoeff, covar = np.polyfit(filtered_wave, filtered_flux_data, order, full = False, cov = True)
            p1, p0 = linecoeff
            std_p1, std_p0 = np.sqrt(covar[0,0]), np.sqrt(covar[1,1])
            flux_cont = np.polyval(linecoeff, w[indrange])

            # estimate the error of flux_cont
            # calculate the residual between the continuum and the polynomial, and then calculate the standard deviation of the residual
            flux_cont_out = np.polyval(linecoeff, filtered_wave)
            cont_resid = filtered_flux_data - flux_cont_out
            flux_cont_err = np.abs(np.nanstd(cont_resid)) * np.ones(len(flux_cont))

        if not sigma_clip_or_not:
            # fit polynomial of second order to the continuum region
            linecoeff, covar = np.polyfit(w[indcont], flux[indcont],order, full = False, cov = True)
            std_p1, std_p0 = np.sqrt(covar[0,0]), np.sqrt(covar[1,1])
            flux_cont = np.polyval(linecoeff, w[indrange])
            # flux_cont_err = np.sqrt((w[indrange] * std_p1)**2 + (std_p0)**2)
            galaxy5 = line_fitting_mc()
            flux_cont_err = galaxy5.flux_cont_MC(100, w[indrange], p0, p1, std_p0, std_p1)
        # divide the flux by the polynomial and put the result in our
        # new flux array
        f[:] = flux[indrange]/np.polyval(linecoeff, w[indrange])
        return w[indrange], f, flux_cont, flux_cont_err

    
    def find_bounds_profile(self,center,sigma,num_sigma=6.5):
        """
        Given the center and standard deviation of a Gaussian, returns the upper and lower bounds for the Gaussian profile.
        """
        ub = center+num_sigma*sigma
        lb = center-num_sigma*sigma
        return np.array([ub,lb])
    ###### ending of functions from line_fitting

    def O2_doublet_low_return(self, wave, spec, espec, sb_region = True):
        '''
        The O2_doublet_low_return function takes four parameters - wave, spec, espec, and sb_region.

        It returns four arrays - v_O2_low, v_O2_low_2, flux_v_O2_low, and err_v_O2_low. 
        These arrays represent the velocity array, flux array (in velocity space), and error array (in velocity space), respectively.

        '''
        # return velocity array, flux array (in velocity space), and error array (in velocity space)
        w_3726, w_3729, w0_3726, w0_3729 = self.wave_dict['[OII]&[OII] 3726&3729']
        ref_O2_low = round(w0_3726)
        ref_O2_low_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_O2_low+spacing) and (ref_O2_low-spacing < wave_c[i]):
                ref_O2_low_lst.append(i)
        med_index_O2_low = np.argsort(ref_O2_low_lst)[len(ref_O2_low_lst)//2]
        ref_index_O2_low = ref_O2_low_lst[med_index_O2_low]
        # if sb_region:
        #     self.wave_O2_low = np.float64(wave_c[ref_index_O2_low-23:ref_index_O2_low+42])
        #     self.flux_O2_low = np.float64(spec_c[ref_index_O2_low-23:ref_index_O2_low+42])
        #     self.err_O2_low = np.float64(espec_c[ref_index_O2_low-23:ref_index_O2_low+42])
        # if not sb_region:
        self.wave_O2_low = np.float64(wave_c[ref_index_O2_low-25:ref_index_O2_low+51])
        self.flux_O2_low = np.float64(spec_c[ref_index_O2_low-25:ref_index_O2_low+51])
        self.err_O2_low = np.float64(espec_c[ref_index_O2_low-25:ref_index_O2_low+51])

        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_3726,1)
        # if sb_region:
        cont_O2_low = [[self.wave_O2_low[0], line_lb],[line_ub, self.wave_O2_low[-1]]]
        # if not sb_region:
        #     cont_O2_low = [[self.wave_O2_low[0], 3772.5],[3780, self.wave_O2_low[-1]]]
        self.cont_f_O2_low, self.cont_f_O2_low_err = self.region_around_line(self.wave_O2_low, self.flux_O2_low, cont_O2_low, order = 1)[-2:]
        self.cont_line_dict['[OII]&[OII] 3726&3729'] = np.array([self.cont_f_O2_low, self.cont_f_O2_low_err])
        self.flux_O2_low = self.flux_O2_low - self.cont_f_O2_low
        self.err_O2_low = np.sqrt((self.err_O2_low)**2 + (self.cont_f_O2_low_err)**2)

        # transform wavelength array to velocity array
        self.v_O2_low = ((self.wave_O2_low / w0_3726) - 1) * self.c
        # transform wavelength array to velocity array
        self.v_O2_low_2 = ((self.wave_O2_low / w0_3729) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_O2_low = self.flux_O2_low * w0_3726 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_O2_low = self.err_O2_low * w0_3726 / self.c

        return (self.v_O2_low, self.v_O2_low_2, self.flux_v_O2_low, self.err_v_O2_low)

    def O1_6300_return(self, wave, spec, espec):
        '''
        The O1_6300_return function also takes four parameters - wave, spec, espec, and sb_region. 
        It returns four arrays - v_6300, flux_v_6300, err_v_6300, and flux_6300_line. These arrays represent the velocity array, 
        flux array (in velocity space), error array (in velocity space), and flux array (in wavelength space) for the [OI] 6300 spectral line.

        '''

        # return velocity array, flux array (in velocity space), and error array (in velocity space)
        w_6300, w0_6300 = self.wave_dict['[OI] 6300']
        ref_6300 = round(w0_6300)
        ref_6300_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_6300+spacing) and (ref_6300-spacing < wave_c[i]):
                ref_6300_lst.append(i)
        med_index_6300 = np.argsort(ref_6300_lst)[len(ref_6300_lst)//2]
        ref_index_6300 = ref_6300_lst[med_index_6300]
        self.wave_6300 = np.float64(wave_c[ref_index_6300-25:ref_index_6300+35])
        self.flux_6300 = np.float64(spec_c[ref_index_6300-25:ref_index_6300+35])
        self.err_6300 = np.float64(espec_c[ref_index_6300-25:ref_index_6300+35])

        # mask significant emission lines
        # self.wave_6300[np.where((self.wave_6300 > 6312.5 * (1. + self.redshift) - 6)&(self.wave_6300 < 6312.5 * (1. + self.redshift) + 6))] = 0
        # self.wave_6300 = np.ma.masked_values(self.wave_6300, 0)
        # mask = self.wave_6300.mask
        # self.flux_6300 = np.ma.masked_array(self.flux_6300, mask=mask).compressed()
        # self.err_6300 = np.ma.masked_array(self.err_6300, mask=mask).compressed()
        # self.wave_6300 = self.wave_6300.compressed()

        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_6300,1.5)
        # if sb_region:
        cont_6300 = [[self.wave_6300[0], line_lb],[line_ub, self.wave_6300[-1]]]
        # if not sb_region:
        #     cont_6300 = [[self.wave_6300[0], 4417],[line_ub, self.wave_6300[-1]]]
        self.cont_f_6300, self.cont_f_6300_err = self.region_around_line(self.wave_6300, self.flux_6300, cont_6300, order = 1)[-2:]
        self.cont_line_dict['[OI] 6300'] = np.array([self.cont_f_6300, self.cont_f_6300_err])
        self.flux_6300 = self.flux_6300 - self.cont_f_6300
        self.err_6300 = np.sqrt((self.err_6300)**2 + (self.cont_f_6300_err)**2)

        # transform wavelength array to velocity array
        self.v_6300 = ((self.wave_6300 / w0_6300) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_6300 = self.flux_6300 * w0_6300 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_6300 = self.err_6300 * w0_6300 / self.c

        return (self.v_6300, self.flux_v_6300, self.err_v_6300)

    def O3_4363_return(self, wave, spec, espec):
        '''
        The O3_4363_return function also takes four parameters - wave, spec, espec, and sb_region. 
        It returns four arrays - v_4363, flux_v_4363, err_v_4363, and flux_4363_line. These arrays represent the velocity array, 
        flux array (in velocity space), error array (in velocity space), and flux array (in wavelength space) for the [OIII] 4363 spectral line.

        '''
        # return velocity array, flux array (in velocity space), and error array (in velocity space)
        w_4363, w0_4363 = self.wave_dict['[OIII] 4363']
        ref_4363 = round(w0_4363)
        ref_4363_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_4363+spacing) and (ref_4363-spacing < wave_c[i]):
                ref_4363_lst.append(i)
        med_index_4363 = np.argsort(ref_4363_lst)[len(ref_4363_lst)//2]
        ref_index_4363 = ref_4363_lst[med_index_4363]
        self.wave_4363 = np.float64(wave_c[ref_index_4363-25:ref_index_4363+50])
        self.flux_4363 = np.float64(spec_c[ref_index_4363-25:ref_index_4363+50])
        self.err_4363 = np.float64(espec_c[ref_index_4363-25:ref_index_4363+50])
        # if not sb_region:
        # self.wave_4363 = np.float64(wave_c[ref_index_4363-16:ref_index_4363+30])
        # self.flux_4363 = np.float64(spec_c[ref_index_4363-16:ref_index_4363+30])
        # self.err_4363 = np.float64(espec_c[ref_index_4363-16:ref_index_4363+30])

        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_4363,1)
        # if sb_region:
        cont_4363 = [[self.wave_4363[0], line_lb],[line_ub, self.wave_4363[-1]]]
        # if not sb_region:
        #     cont_4363 = [[self.wave_4363[0], 4417],[line_ub, self.wave_4363[-1]]]
        self.cont_f_4363, self.cont_f_4363_err = self.region_around_line(self.wave_4363, self.flux_4363, cont_4363, order = 1)[-2:]
        self.cont_line_dict['[OIII] 4363'] = np.array([self.cont_f_4363, self.cont_f_4363_err])
        self.flux_4363 = self.flux_4363 - self.cont_f_4363
        self.err_4363 = np.sqrt((self.err_4363)**2 + (self.cont_f_4363_err)**2)

        # transform wavelength array to velocity array
        self.v_4363 = ((self.wave_4363 / w0_4363) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_4363 = self.flux_4363 * w0_4363 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_4363 = self.err_4363 * w0_4363 / self.c

        return (self.v_4363, self.flux_v_4363, self.err_v_4363)


    def O3_4959_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line [OIII] 4959.
        '''
        # return velocity array, flux array (in velocity space), and error array (in velocity space)
        w_4959, w0_4959 = self.wave_dict['[OIII] 4959']
        ref_4959 = round(w0_4959)
        ref_4959_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_4959+spacing) and (ref_4959-spacing < wave_c[i]):
                ref_4959_lst.append(i)
        med_index_4959 = np.argsort(ref_4959_lst)[len(ref_4959_lst)//2]
        ref_index_4959 = ref_4959_lst[med_index_4959]
        self.wave_4959 = np.float64(wave_c[ref_index_4959-100:ref_index_4959+45])
        self.flux_4959 = np.float64(spec_c[ref_index_4959-100:ref_index_4959+45])
        self.err_4959 = np.float64(espec_c[ref_index_4959-100:ref_index_4959+45])

        # mask significant emission lines
        self.wave_4959[np.where((self.wave_4959 > 4983)&(self.wave_4959 < 4988.5))] = 0
        self.wave_4959 = np.ma.masked_values(self.wave_4959, 0)
        mask = self.wave_4959.mask
        self.flux_4959 = np.ma.masked_array(self.flux_4959, mask=mask).compressed()
        self.err_4959 = np.ma.masked_array(self.err_4959, mask=mask).compressed()
        self.wave_4959 = self.wave_4959.compressed()

        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_4959,3)
        cont_4959 = [[self.wave_4959[0], line_lb],[line_ub, self.wave_4959[-1]]]
        self.cont_f_4959, self.cont_f_4959_err = self.region_around_line(self.wave_4959, self.flux_4959, cont_4959, order = 1)[-2:]
        self.cont_line_dict['[OIII] 4959'] = np.array([self.cont_f_4959, self.cont_f_4959_err])
        self.flux_4959 = self.flux_4959 - self.cont_f_4959
        self.err_4959 = np.sqrt((self.err_4959)**2 + (self.cont_f_4959_err)**2)

        # transform wavelength array to velocity array
        self.v_4959 = ((self.wave_4959 / w0_4959) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_4959 = self.flux_4959 * w0_4959 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_4959 = self.err_4959 * w0_4959 / self.c

        return (self.v_4959, self.flux_v_4959, self.err_v_4959)

    def O3_5007_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line [OIII] 5007.
        '''
        # return velocity array, flux array (in velocity space), and error array (in velocity space)
        w_5007, w0_5007 = self.wave_dict['[OIII] 5007']
        ref_5007 = round(w0_5007)
        ref_5007_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_5007+spacing) and (ref_5007-spacing < wave_c[i]):
                ref_5007_lst.append(i)
        med_index_5007 = np.argsort(ref_5007_lst)[len(ref_5007_lst)//2]
        ref_index_5007 = ref_5007_lst[med_index_5007]
        self.wave_5007 = np.float64(wave_c[ref_index_5007-65:ref_index_5007+65])
        self.flux_5007 = np.float64(spec_c[ref_index_5007-65:ref_index_5007+65])
        self.err_5007 = np.float64(espec_c[ref_index_5007-65:ref_index_5007+65])

        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_5007, 3.5)
        cont_5007 = [[self.wave_5007[0], line_lb],[line_ub, self.wave_5007[-1]]]
        self.cont_f_5007, self.cont_f_5007_err = self.region_around_line(self.wave_5007, self.flux_5007, cont_5007, order = 1)[-2:]
        self.cont_line_dict['[OIII] 5007'] = np.array([self.cont_f_5007, self.cont_f_5007_err])
        self.flux_5007 = self.flux_5007 - self.cont_f_5007
        self.err_5007 = np.sqrt((self.err_5007)**2 + (self.cont_f_5007_err)**2)

        # transform wavelength array to velocity array
        self.v_5007 = ((self.wave_5007 / w0_5007) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_5007 = self.flux_5007 * w0_5007 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_5007 = self.err_5007 * w0_5007 / self.c

        return (self.v_5007, self.flux_v_5007, self.err_v_5007)


    def O3_5007_HeI_5015_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line [OIII] 5007.
        '''
        # return velocity array, flux array (in velocity space), and error array (in velocity space)
        w_5007, w_5015, w0_5007, w0_5015 = self.wave_dict['[OIII]&HeI 5007&5015']
        ref_5007_5015 = round(w0_5007)
        ref_5007_5015_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_5007_5015+spacing) and (ref_5007_5015-spacing < wave_c[i]):
                ref_5007_5015_lst.append(i)
        med_index_5007_5015 = np.argsort(ref_5007_5015_lst)[len(ref_5007_5015_lst)//2]
        ref_index_5007_5015 = ref_5007_5015_lst[med_index_5007_5015]
        self.wave_5007_5015 = np.float64(wave_c[ref_index_5007_5015-65:ref_index_5007_5015+65])
        self.flux_5007_5015 = np.float64(spec_c[ref_index_5007_5015-65:ref_index_5007_5015+65])
        self.err_5007_5015 = np.float64(espec_c[ref_index_5007_5015-65:ref_index_5007_5015+65])

        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_5007, 3.7)
        cont_5007_5015 = [[self.wave_5007_5015[0], line_lb],[line_ub, self.wave_5007_5015[-1]]]
        self.cont_f_5007_5015, self.cont_f_5007_5015_err = self.region_around_line(self.wave_5007_5015, self.flux_5007_5015, cont_5007_5015, order = 1)[-2:]
        self.cont_line_dict['[OIII]&HeI 5007&5015'] = np.array([self.cont_f_5007_5015, self.cont_f_5007_5015_err])
        self.flux_5007_5015 = self.flux_5007_5015 - self.cont_f_5007_5015
        self.err_5007_5015 = np.sqrt((self.err_5007_5015)**2 + (self.cont_f_5007_5015_err)**2)

        # transform wavelength array to velocity array
        self.v_5007_5015 = ((self.wave_5007_5015 / w0_5007) - 1) * self.c
        # transform wavelength array to velocity array
        self.v_5007_5015_2 = ((self.wave_5007_5015 / w0_5015) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_5007_5015 = self.flux_5007_5015 * w0_5007 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_5007_5015 = self.err_5007_5015 * w0_5007 / self.c

        return (self.v_5007_5015, self.v_5007_5015_2, self.flux_v_5007_5015, self.err_v_5007_5015)


    def Hv_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line Hv.
        '''
        w_Hv, w0_Hv = self.wave_dict['H nu']
        ref_Hv = round(w0_Hv)
        ref_Hv_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_Hv+spacing) and (ref_Hv-spacing < wave_c[i]):
                ref_Hv_lst.append(i)
        med_index_Hv = np.argsort(ref_Hv_lst)[len(ref_Hv_lst)//2]
        ref_index_Hv = ref_Hv_lst[med_index_Hv]
        self.wave_Hv = np.float64(wave_c[ref_index_Hv-8:ref_index_Hv+11])
        self.flux_Hv = np.float64(spec_c[ref_index_Hv-8:ref_index_Hv+11])
        self.err_Hv = np.float64(espec_c[ref_index_Hv-8:ref_index_Hv+11])

        # determine the local continuum and subtract it from the flux array
        if 'H nu' not in self.absorption_lines:
            sigma_cont = 1
        if 'H nu' in self.absorption_lines:
            sigma_cont = 3.4
        line_ub, line_lb = self.find_bounds_profile(w0_Hv,sigma_cont,num_sigma=3.5)
        cont_Hv = [[self.wave_Hv[0], 3749.5],[line_ub, self.wave_Hv[-1]]]
        self.cont_f_Hv, self.cont_f_Hv_err = self.region_around_line(self.wave_Hv, self.flux_Hv, cont_Hv, order = 1)[-2:]
        self.cont_line_dict['H nu'] = np.array([self.cont_f_Hv, self.cont_f_Hv_err])
        self.flux_Hv = self.flux_Hv - self.cont_f_Hv
        self.err_Hv = np.sqrt((self.err_Hv)**2 + (self.cont_f_Hv_err)**2)

        # transform wavelength array to velocity array
        self.v_Hv = ((self.wave_Hv / w0_Hv) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_Hv = self.flux_Hv * w0_Hv / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_Hv = self.err_Hv * w0_Hv / self.c

        return (self.v_Hv, self.flux_v_Hv, self.err_v_Hv)


    def Hk_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line Hk.
        '''
        w_Hk, w0_Hk = self.wave_dict['H kappa']
        ref_Hk = round(w0_Hk)
        ref_Hk_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_Hk+spacing) and (ref_Hk-spacing < wave_c[i]):
                ref_Hk_lst.append(i)
        med_index_Hk = np.argsort(ref_Hk_lst)[len(ref_Hk_lst)//2]
        ref_index_Hk = ref_Hk_lst[med_index_Hk]
        self.wave_Hk = np.float64(wave_c[ref_index_Hk-20:ref_index_Hk+22])
        self.flux_Hk = np.float64(spec_c[ref_index_Hk-20:ref_index_Hk+22])
        self.err_Hk = np.float64(espec_c[ref_index_Hk-20:ref_index_Hk+22])

        # determine the local continuum and subtract it from the flux array
        if 'H kappa' not in self.absorption_lines:
            sigma_cont = 1
        if 'H kappa' in self.absorption_lines:
            sigma_cont = 3.4
        line_ub, line_lb = self.find_bounds_profile(w0_Hk,sigma_cont,num_sigma=4)
        cont_Hk = [[self.wave_Hk[0], line_lb],[line_ub, self.wave_Hk[-1]]]
        self.cont_f_Hk, self.cont_f_Hk_err = self.region_around_line(self.wave_Hk, self.flux_Hk, cont_Hk, order = 1)[-2:]
        self.cont_line_dict['H kappa'] = np.array([self.cont_f_Hk, self.cont_f_Hk_err])
        self.flux_Hk = self.flux_Hk - self.cont_f_Hk
        self.err_Hk = np.sqrt((self.err_Hk)**2 + (self.cont_f_Hk_err)**2)

        # transform wavelength array to velocity array
        self.v_Hk = ((self.wave_Hk / w0_Hk) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_Hk = self.flux_Hk * w0_Hk / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_Hk = self.err_Hk * w0_Hk / self.c

        return (self.v_Hk, self.flux_v_Hk, self.err_v_Hk)


    def Hi_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line Hi.
        '''
        w_Hi, w0_Hi = self.wave_dict['H iota']
        ref_Hi = round(w0_Hi)
        ref_Hi_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_Hi+spacing) and (ref_Hi-spacing < wave_c[i]):
                ref_Hi_lst.append(i)
        med_index_Hi = np.argsort(ref_Hi_lst)[len(ref_Hi_lst)//2]
        ref_index_Hi = ref_Hi_lst[med_index_Hi]
        self.wave_Hi = np.float64(wave_c[ref_index_Hi-23:ref_index_Hi+25])
        self.flux_Hi = np.float64(spec_c[ref_index_Hi-23:ref_index_Hi+25])
        self.err_Hi = np.float64(espec_c[ref_index_Hi-23:ref_index_Hi+25])

        # determine the local continuum and subtract it from the flux array
        if 'H iota' not in self.absorption_lines:
            sigma_cont = 1
        if 'H iota' in self.absorption_lines:
            sigma_cont = 3.4
        line_ub, line_lb = self.find_bounds_profile(w0_Hi,sigma_cont,num_sigma=6.5)
        cont_Hi = [[self.wave_Hi[0], line_lb],[line_ub, self.wave_Hi[-1]]]
        self.cont_f_Hi, self.cont_f_Hi_err = self.region_around_line(self.wave_Hi, self.flux_Hi, cont_Hi, order = 1)[-2:]
        self.cont_line_dict['H iota'] = np.array([self.cont_f_Hi, self.cont_f_Hi_err])
        self.flux_Hi = self.flux_Hi - self.cont_f_Hi
        self.err_Hi = np.sqrt((self.err_Hi)**2 + (self.cont_f_Hi_err)**2)

        # transform wavelength array to velocity array
        self.v_Hi = ((self.wave_Hi / w0_Hi) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_Hi = self.flux_Hi * w0_Hi / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_Hi = self.err_Hi * w0_Hi / self.c

        return (self.v_Hi, self.flux_v_Hi, self.err_v_Hi)


    def Hd_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line H-delta.
        '''
        w_Hd, w0_Hd = self.wave_dict['H delta']
        ref_Hd = round(w0_Hd)
        ref_Hd_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_Hd+spacing) and (ref_Hd-spacing < wave_c[i]):
                ref_Hd_lst.append(i)
        med_index_Hd = np.argsort(ref_Hd_lst)[len(ref_Hd_lst)//2]
        ref_index_Hd = ref_Hd_lst[med_index_Hd]
        self.wave_Hd = np.float64(wave_c[ref_index_Hd-100:ref_index_Hd+100])
        self.flux_Hd = np.float64(spec_c[ref_index_Hd-100:ref_index_Hd+100])
        self.err_Hd = np.float64(espec_c[ref_index_Hd-100:ref_index_Hd+100])

        # determine the local continuum and subtract it from the flux array
        if 'H delta' not in self.absorption_lines:
            sigma_cont = 2
        if 'H delta' in self.absorption_lines:
            sigma_cont = 3.7
        line_ub, line_lb = self.find_bounds_profile(w0_Hd,sigma_cont)
        cont_Hd = [[self.wave_Hd[0], line_lb],[line_ub, self.wave_Hd[-1]]]

        self.cont_f_Hd, self.cont_f_Hd_err = self.region_around_line(self.wave_Hd, self.flux_Hd, cont_Hd, order = 1)[-2:]
        self.cont_line_dict['H delta'] = np.array([self.cont_f_Hd, self.cont_f_Hd_err])
        self.flux_Hd = self.flux_Hd - self.cont_f_Hd
        self.err_Hd = np.sqrt((self.err_Hd)**2 + (self.cont_f_Hd_err)**2)

        # transform wavelength array to velocity array
        self.v_Hd = ((self.wave_Hd / w0_Hd) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_Hd = self.flux_Hd * w0_Hd / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_Hd = self.err_Hd * w0_Hd / self.c

        return (self.v_Hd, self.flux_v_Hd, self.err_v_Hd)


    def Hy_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line H-gamma.
        '''
        w_Hy, w0_Hy = self.wave_dict['H gamma']
        ref_Hy = round(w0_Hy)
        ref_Hy_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_Hy+spacing) and (ref_Hy-spacing < wave_c[i]):
                ref_Hy_lst.append(i)
        med_index_Hy = np.argsort(ref_Hy_lst)[len(ref_Hy_lst)//2]
        ref_index_Hy = ref_Hy_lst[med_index_Hy]
        # determine the local continuum and subtract it from the flux array
        if 'H gamma' not in self.absorption_lines:
            self.wave_Hy = np.float64(wave_c[ref_index_Hy-70:ref_index_Hy+70])
            self.flux_Hy = np.float64(spec_c[ref_index_Hy-70:ref_index_Hy+70])
            self.err_Hy = np.float64(espec_c[ref_index_Hy-70:ref_index_Hy+70])
            sigma_cont = 1.2
        if 'H gamma' in self.absorption_lines:
            self.wave_Hy = np.float64(wave_c[ref_index_Hy-100:ref_index_Hy+100])
            self.flux_Hy = np.float64(spec_c[ref_index_Hy-100:ref_index_Hy+100])
            self.err_Hy = np.float64(espec_c[ref_index_Hy-100:ref_index_Hy+100])
            sigma_cont = 3.4
        line_ub, line_lb = self.find_bounds_profile(w0_Hy,sigma_cont)
        cont_Hy = [[self.wave_Hy[0], line_lb],[line_ub, self.wave_Hy[-1]]]

        self.cont_f_Hy, self.cont_f_Hy_err = self.region_around_line(self.wave_Hy, self.flux_Hy, cont_Hy, order = 1)[-2:]
        self.cont_line_dict['H gamma'] = np.array([self.cont_f_Hy, self.cont_f_Hy_err])
        self.flux_Hy = self.flux_Hy - self.cont_f_Hy
        self.err_Hy = np.sqrt((self.err_Hy)**2 + (self.cont_f_Hy_err)**2)

        # transform wavelength array to velocity array
        self.v_Hy = ((self.wave_Hy / w0_Hy) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_Hy = self.flux_Hy * w0_Hy / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_Hy = self.err_Hy * w0_Hy / self.c

        return (self.v_Hy, self.flux_v_Hy, self.err_v_Hy)


    def Hb_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line H-beta line.
        '''
        w_Hb, w0_Hb = self.wave_dict['H beta']
        ref_Hb = round(w0_Hb)
        ref_Hb_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_Hb+spacing) and (ref_Hb-spacing < wave_c[i]):
                ref_Hb_lst.append(i)
        med_index_Hb = np.argsort(ref_Hb_lst)[len(ref_Hb_lst)//2]
        ref_index_Hb = ref_Hb_lst[med_index_Hb]
        self.wave_Hb = np.float64(wave_c[ref_index_Hb-70:ref_index_Hb+70])
        self.flux_Hb = np.float64(spec_c[ref_index_Hb-70:ref_index_Hb+70])
        self.err_Hb = np.float64(espec_c[ref_index_Hb-70:ref_index_Hb+70])

        # determine the local continuum and subtract it from the flux array
        if 'H beta' not in self.absorption_lines:
            sigma_cont = 2
        if 'H beta' in self.absorption_lines:
            sigma_cont = 3.4
        line_ub, line_lb = self.find_bounds_profile(w0_Hb,sigma_cont)
        cont_Hb = [[self.wave_Hb[0], line_lb],[line_ub, self.wave_Hb[-1]]]

        self.cont_f_Hb, self.cont_f_Hb_err = self.region_around_line(self.wave_Hb, self.flux_Hb, cont_Hb, order = 1)[-2:]
        self.cont_line_dict['H beta'] = np.array([self.cont_f_Hb, self.cont_f_Hb_err])
        self.flux_Hb = self.flux_Hb - self.cont_f_Hb
        self.err_Hb = np.sqrt((self.err_Hb)**2 + (self.cont_f_Hb_err)**2)

        # transform wavelength array to velocity array
        self.v_Hb = ((self.wave_Hb / w0_Hb) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_Hb = self.flux_Hb * w0_Hb / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_Hb = self.err_Hb * w0_Hb / self.c

        return (self.v_Hb, self.flux_v_Hb, self.err_v_Hb)

    def Ha_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line H-alpha line.
        '''
        w_Ha, w0_Ha = self.wave_dict['H alpha']
        ref_Ha = round(w0_Ha)
        ref_Ha_lst = []
        spacing = self.max_spacing(wave)

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_Ha+spacing) and (ref_Ha-spacing < wave_c[i]):
                ref_Ha_lst.append(i)
        med_index_Ha = np.argsort(ref_Ha_lst)[len(ref_Ha_lst)//2]
        ref_index_Ha = ref_Ha_lst[med_index_Ha]
        self.wave_Ha = np.float64(wave_c[ref_index_Ha-70:ref_index_Ha+70])
        self.flux_Ha = np.float64(spec_c[ref_index_Ha-70:ref_index_Ha+70])
        self.err_Ha = np.float64(espec_c[ref_index_Ha-70:ref_index_Ha+70])

        # determine the local continuum and subtract it from the flux array
        if 'H alpha' not in self.absorption_lines:
            sigma_cont = 2
        if 'H alpha' in self.absorption_lines:
            sigma_cont = 3.4
        line_ub, line_lb = self.find_bounds_profile(w0_Ha,sigma_cont)
        cont_Ha = [[self.wave_Ha[0], line_lb],[line_ub, self.wave_Ha[-1]]]

        self.cont_f_Ha, self.cont_f_Ha_err = self.region_around_line(self.wave_Ha, self.flux_Ha, cont_Ha, order = 1)[-2:]
        self.cont_line_dict['H alpha'] = np.array([self.cont_f_Ha, self.cont_f_Ha_err])
        self.flux_Ha = self.flux_Ha - self.cont_f_Ha
        self.err_Ha = np.sqrt((self.err_Ha)**2 + (self.cont_f_Ha_err)**2)

        # transform wavelength array to velocity array
        self.v_Ha = ((self.wave_Ha / w0_Ha) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_Ha = self.flux_Ha * w0_Ha / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_Ha = self.err_Ha * w0_Ha / self.c

        return (self.v_Ha, self.flux_v_Ha, self.err_v_Ha)

    def N2_6548_Ha_N2_6583_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission lines [NII] 6548, H alpha, and [NII] 6583 (blended with each other).
        '''
        # return velocity array, flux array (in velocity space), and error array (in velocity space)
        w_6548, w_Ha, w_6583, w0_6548, w0_Ha, w0_6583 = self.wave_dict['[NII]&H&[NII] 6548&alpha&6583']
        ref_6548_Ha_6583 = round(w0_Ha)
        ref_6548_Ha_6583_lst = []
        spacing = 5.

        # get rid of nan values in spec and espec arrays
        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])
        
        # extract a narrow band around this specific line
        for i in range(len(wave_c)):
            if (wave_c[i]<ref_6548_Ha_6583+spacing) and (ref_6548_Ha_6583-spacing < wave_c[i]):
                ref_6548_Ha_6583_lst.append(i)
        med_index_6548_Ha_6583 = np.argsort(ref_6548_Ha_6583_lst)[len(ref_6548_Ha_6583_lst)//2]
        ref_index_6548_Ha_6583 = ref_6548_Ha_6583_lst[med_index_6548_Ha_6583]
        self.wave_6548_Ha_6583 = np.float64(wave_c[ref_index_6548_Ha_6583-70:ref_index_6548_Ha_6583+70])
        self.flux_6548_Ha_6583 = np.float64(spec_c[ref_index_6548_Ha_6583-70:ref_index_6548_Ha_6583+70])
        self.err_6548_Ha_6583 = np.float64(espec_c[ref_index_6548_Ha_6583-70:ref_index_6548_Ha_6583+70])

        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_Ha, 3.5)
        cont_6548_Ha_6583 = [[self.wave_6548_Ha_6583[0], line_lb],[line_ub, self.wave_6548_Ha_6583[-1]]]
        self.cont_f_6548_Ha_6583, self.cont_f_6548_Ha_6583_err = self.region_around_line(self.wave_6548_Ha_6583, self.flux_6548_Ha_6583, cont_6548_Ha_6583, order = 1)[-2:]
        self.cont_line_dict['[NII]&H&[NII] 6548&alpha&6583'] = np.array([self.cont_f_6548_Ha_6583, self.cont_f_6548_Ha_6583_err])
        self.flux_6548_Ha_6583 = self.flux_6548_Ha_6583 - self.cont_f_6548_Ha_6583
        self.err_6548_Ha_6583 = np.sqrt((self.err_6548_Ha_6583)**2 + (self.cont_f_6548_Ha_6583_err)**2)

        # transform wavelength array to velocity array
        self.v_6548_Ha_6583 = ((self.wave_6548_Ha_6583 / w0_6548) - 1) * self.c
        # transform wavelength array to velocity array
        self.v_6548_Ha_6583_2 = ((self.wave_6548_Ha_6583 / w0_Ha) - 1) * self.c
        # transform wavelength array to velocity array
        self.v_6548_Ha_6583_3 = ((self.wave_6548_Ha_6583 / w0_6583) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_6548_Ha_6583 = self.flux_6548_Ha_6583 * w0_Ha / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_6548_Ha_6583 = self.err_6548_Ha_6583 * w0_Ha / self.c

        return (self.v_6548_Ha_6583, self.v_6548_Ha_6583_2, self.v_6548_Ha_6583_3, self.flux_v_6548_Ha_6583, self.err_v_6548_Ha_6583)

    def S2_doublet_return(self, wave, spec, espec, sb_region = True):
        '''
        The S2_doublet_return function takes four parameters - wave, spec, espec, and sb_region.

        It returns four arrays - v_S2, v_S2_2, flux_v_S2, and err_v_S2. 
        These arrays represent the velocity array, flux array (in velocity space), and error array (in velocity space), respectively.

        '''
        # return velocity array, flux array (in velocity space), and error array (in velocity space)
        w_6716, w_6731, w0_6716, w0_6731 = self.wave_dict['[SII]&[SII] 6716&6731']
        ref_S2 = round(w0_6716)
        ref_S2_lst = []
        spacing = 5.

        # get rid of nan values in spec and espec arrays
        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        # extract a narrow band around this specific line
        for i in range(len(wave_c)):
            if (wave_c[i]<ref_S2+spacing) and (ref_S2-spacing < wave_c[i]):
                ref_S2_lst.append(i)
        med_index_S2 = np.argsort(ref_S2_lst)[len(ref_S2_lst)//2]
        ref_index_S2 = ref_S2_lst[med_index_S2]
        self.wave_S2 = np.float64(wave_c[ref_index_S2-30:ref_index_S2+51])
        self.flux_S2 = np.float64(spec_c[ref_index_S2-30:ref_index_S2+51])
        self.err_S2 = np.float64(espec_c[ref_index_S2-30:ref_index_S2+51])

        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_6716, 2)
        # if sb_region:
        cont_S2 = [[self.wave_S2[0], line_lb],[line_ub, self.wave_S2[-1]]]
        # if not sb_region:
        #     cont_S2 = [[self.wave_S2[0], 3772.5],[3780, self.wave_S2[-1]]]
        self.cont_f_S2, self.cont_f_S2_err = self.region_around_line(self.wave_S2, self.flux_S2, cont_S2, order = 1)[-2:]
        self.cont_line_dict['[SII]&[SII] 6716&6731'] = np.array([self.cont_f_S2, self.cont_f_S2_err])
        self.flux_S2 = self.flux_S2 - self.cont_f_S2
        self.err_S2 = np.sqrt((self.err_S2)**2 + (self.cont_f_S2_err)**2)

        # transform wavelength array to velocity array
        self.v_S2 = ((self.wave_S2 / w0_6716) - 1) * self.c
        # transform wavelength array to velocity array
        self.v_S2_2 = ((self.wave_S2 / w0_6731) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_S2 = self.flux_S2 * w0_6716 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_S2 = self.err_S2 * w0_6716 / self.c

        return (self.v_S2, self.v_S2_2, self.flux_v_S2, self.err_v_S2)

    def He1_5015_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line HeI 5015.
        '''
        w_5015, w0_5015 = self.wave_dict['HeI 5015']
        ref_5015 = round(w0_5015)
        ref_5015_lst = []
        spacing = 5.
        
        # get rid of nan values in spec and espec arrays
        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        # extract a narrow band around this specific line
        for i in range(len(wave_c)):
            if (wave_c[i]<ref_5015+spacing) and (ref_5015-spacing < wave_c[i]):
                ref_5015_lst.append(i)
        med_index_5015 = np.argsort(ref_5015_lst)[len(ref_5015_lst)//2]
        ref_index_5015 = ref_5015_lst[med_index_5015]
        self.wave_5015 = np.float64(wave_c[ref_index_5015-15:ref_index_5015+15])
        self.flux_5015 = np.float64(spec_c[ref_index_5015-15:ref_index_5015+15])
        self.err_5015 = np.float64(espec_c[ref_index_5015-15:ref_index_5015+15])
   
        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_5015,1)
        cont_5015 = [[self.wave_5015[0], line_lb],[line_ub, self.wave_5015[-1]]]

        self.cont_f_5015, self.cont_f_5015_err = self.region_around_line(self.wave_5015, self.flux_5015, cont_5015, order = 1)[-2:]
        self.cont_line_dict['HeI 5015'] = np.array([self.cont_f_5015, self.cont_f_5015_err])
        self.flux_5015 = self.flux_5015 - self.cont_f_5015
        self.err_5015 = np.sqrt((self.err_5015)**2 + (self.cont_f_5015_err)**2)

        # transform wavelength array to velocity array
        self.v_5015 = ((self.wave_5015 / w0_5015) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_5015 = self.flux_5015 * w0_5015 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_5015 = self.err_5015 * w0_5015 / self.c

        return (self.v_5015, self.flux_v_5015, self.err_v_5015)

    def He2_4686_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line HeII 4686.
        '''
        w_4686, w0_4686 = self.wave_dict['HeII 4686']
        ref_4686 = round(w0_4686)
        ref_4686_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_4686+spacing) and (ref_4686-spacing < wave_c[i]):
                ref_4686_lst.append(i)
        med_index_4686 = np.argsort(ref_4686_lst)[len(ref_4686_lst)//2]
        ref_index_4686 = ref_4686_lst[med_index_4686]
        self.wave_4686 = np.float64(wave_c[ref_index_4686-25:ref_index_4686+25])
        self.flux_4686 = np.float64(spec_c[ref_index_4686-25:ref_index_4686+25])
        self.err_4686 = np.float64(espec_c[ref_index_4686-25:ref_index_4686+25])
   
        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_4686,1)
        cont_4686 = [[self.wave_4686[0], line_lb],[line_ub, self.wave_4686[-1]]]

        self.cont_f_4686, self.cont_f_4686_err = self.region_around_line(self.wave_4686, self.flux_4686, cont_4686, order = 1)[-2:]
        self.cont_line_dict['HeII 4686'] = np.array([self.cont_f_4686, self.cont_f_4686_err])
        self.flux_4686 = self.flux_4686 - self.cont_f_4686
        self.err_4686 = np.sqrt((self.err_4686)**2 + (self.cont_f_4686_err)**2)

        # transform wavelength array to velocity array
        self.v_4686 = ((self.wave_4686 / w0_4686) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_4686 = self.flux_4686 * w0_4686 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_4686 = self.err_4686 * w0_4686 / self.c

        return (self.v_4686, self.flux_v_4686, self.err_v_4686)

    def Ar4_4711_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line ArIV 4711.
        '''
        w_4711, w0_4711 = self.wave_dict['[ArIV] 4711']
        ref_4711 = round(w0_4711)
        ref_4711_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_4711+spacing) and (ref_4711-spacing < wave_c[i]):
                ref_4711_lst.append(i)
        med_index_4711 = np.argsort(ref_4711_lst)[len(ref_4711_lst)//2]
        ref_index_4711 = ref_4711_lst[med_index_4711]
        self.wave_4711 = np.float64(wave_c[ref_index_4711-25:ref_index_4711+25])
        self.flux_4711 = np.float64(spec_c[ref_index_4711-25:ref_index_4711+25])
        self.err_4711 = np.float64(espec_c[ref_index_4711-25:ref_index_4711+25])
   
        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_4711,1)
        cont_4711 = [[self.wave_4711[0], line_lb],[line_ub, self.wave_4711[-1]]]

        self.cont_f_4711, self.cont_f_4711_err = self.region_around_line(self.wave_4711, self.flux_4711, cont_4711, order = 1)[-2:]
        self.cont_line_dict['[ArIV] 4711'] = np.array([self.cont_f_4711, self.cont_f_4711_err])
        self.flux_4711 = self.flux_4711 - self.cont_f_4711
        self.err_4711 = np.sqrt((self.err_4711)**2 + (self.cont_f_4711_err)**2)

        # transform wavelength array to velocity array
        self.v_4711 = ((self.wave_4711 / w0_4711) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_4711 = self.flux_4711 * w0_4711 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_4711 = self.err_4711 * w0_4711 / self.c

        return (self.v_4711, self.flux_v_4711, self.err_v_4711)

    def Ar4_4711_HeI_4713_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission lines [ArIV] 4711 & HeI 4713.
        '''
        # return velocity array, flux array (in velocity space), and error array (in velocity space)
        w_4711, w_4713, w0_4711, w0_4713 = self.wave_dict['[ArIV]&HeI 4711&4713']
        ref_4711_4713 = round(w0_4711)
        ref_4711_4713_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_4711_4713+spacing) and (ref_4711_4713-spacing < wave_c[i]):
                ref_4711_4713_lst.append(i)
        med_index_4711_4713 = np.argsort(ref_4711_4713_lst)[len(ref_4711_4713_lst)//2]
        ref_index_4711_4713 = ref_4711_4713_lst[med_index_4711_4713]
        self.wave_4711_4713 = np.float64(wave_c[ref_index_4711_4713-30:ref_index_4711_4713+30])
        self.flux_4711_4713 = np.float64(spec_c[ref_index_4711_4713-30:ref_index_4711_4713+30])
        self.err_4711_4713 = np.float64(espec_c[ref_index_4711_4713-30:ref_index_4711_4713+30])

        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_4711, 1.5)
        cont_4711_4713 = [[self.wave_4711_4713[0], line_lb],[line_ub, self.wave_4711_4713[-1]]]
        self.cont_f_4711_4713, self.cont_f_4711_4713_err = self.region_around_line(self.wave_4711_4713, self.flux_4711_4713, cont_4711_4713, order = 1)[-2:]
        self.cont_line_dict['[ArIV]&HeI 4711&4713'] = np.array([self.cont_f_4711_4713, self.cont_f_4711_4713_err])
        self.flux_4711_4713 = self.flux_4711_4713 - self.cont_f_4711_4713
        self.err_4711_4713 = np.sqrt((self.err_4711_4713)**2 + (self.cont_f_4711_4713_err)**2)

        # transform wavelength array to velocity array
        self.v_4711_4713 = ((self.wave_4711_4713 / w0_4711) - 1) * self.c
        # transform wavelength array to velocity array
        self.v_4711_4713_2 = ((self.wave_4711_4713 / w0_4713) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_4711_4713 = self.flux_4711_4713 * w0_4711 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_4711_4713 = self.err_4711_4713 * w0_4711 / self.c

        return (self.v_4711_4713, self.v_4711_4713_2, self.flux_v_4711_4713, self.err_v_4711_4713)

    def Ar4_4740_return(self, wave, spec, espec):
        '''
        The method take in three arguments, wave, spec, and espec, which are arrays representing the wavelength, spectrum and error values for a given spectrum.

        It returns the velocity array, flux array (in velocity space), and error array (in velocity space) for the emission line ArIV 4740.
        '''
        w_4740, w0_4740 = self.wave_dict['[ArIV] 4740']
        ref_4740 = round(w0_4740)
        ref_4740_lst = []
        spacing = 5.

        nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        for i in range(len(wave_c)):
            if (wave_c[i]<ref_4740+spacing) and (ref_4740-spacing < wave_c[i]):
                ref_4740_lst.append(i)
        med_index_4740 = np.argsort(ref_4740_lst)[len(ref_4740_lst)//2]
        ref_index_4740 = ref_4740_lst[med_index_4740]
        self.wave_4740 = np.float64(wave_c[ref_index_4740-25:ref_index_4740+25])
        self.flux_4740 = np.float64(spec_c[ref_index_4740-25:ref_index_4740+25])
        self.err_4740 = np.float64(espec_c[ref_index_4740-25:ref_index_4740+25])
   
        # determine the local continuum and subtract it from the flux array
        line_ub, line_lb = self.find_bounds_profile(w0_4740,1)
        cont_4740 = [[self.wave_4740[0], line_lb],[line_ub, self.wave_4740[-1]]]

        self.cont_f_4740, self.cont_f_4740_err = self.region_around_line(self.wave_4740, self.flux_4740, cont_4740, order = 1)[-2:]
        self.cont_line_dict['[ArIV] 4740'] = np.array([self.cont_f_4740, self.cont_f_4740_err])
        self.flux_4740 = self.flux_4740 - self.cont_f_4740
        self.err_4740 = np.sqrt((self.err_4740)**2 + (self.cont_f_4740_err)**2)

        # transform wavelength array to velocity array
        self.v_4740 = ((self.wave_4740 / w0_4740) - 1) * self.c
        # transform continuum-subtracted flux array to velocity space
        self.flux_v_4740 = self.flux_4740 * w0_4740 / self.c
        # transform continuum-subtracted err array to velocity space
        self.err_v_4740 = self.err_4740 * w0_4740 / self.c

        return (self.v_4740, self.flux_v_4740, self.err_v_4740)


    def all_lines_result(self, wave, spec, espec, n_iteration = 1000, get_flux = False, get_corr = True, save_flux_table = False,
                         get_ew = True, save_ew_table = False, get_error = False, save_par_table = False):
        """
        Perform a simultaneous line-fitting algorithm on various emission lines in a given spectrum. Return the best-fitting parameters of a n_iteration fitting.
        
        Parameters:
        -----------
        wave : array_like
            1D array of the wavelength values of the spectrum.
        spec : array_like
            1D array of the flux values of the spectrum.
        espec : array_like
            1D array of the flux errors of the spectrum.
        n_iteration : int, optional
            Number of iterations to run the fitting. Default is 1000.
        get_flux : bool, optional
            If True, also return the best-fitting flux for each line. Default is False.
        get_corr : bool, optional
            If True, also return the best-fitting corrected flux for each line. Default is True.
        save_flux_table : bool, optional
            If True, save the best-fitting flux pandas table for each line. Default is False.
        get_ew : bool, optional
            If True, also print the best-fitting equivalent width for each line. Default is False.
        save_ew_table : bool, optional
            If True, save the best-fitting equivalent width pandas table for each line. Default is False.
        get_error : bool, optional
            If True, also calculate the errors on the best-fitting parameters. Default is False.
        save_par_table : bool, optional
            If True, save the best-fitting velocity width pandas table for each velocity component. Default is False.
        Returns:
        --------
        """
        # return the best-fitting models
        self.model_dict, self.best_chi2 = self.fitting_input_params(wave, spec, espec, n_iteration = n_iteration)

        # best-fitting velocity centers and widths
        self.x0_e = self.galaxy4.best_param_dict["center_e"]
        self.sigma_e = self.galaxy4.best_param_dict["sigma_e"]
        self.x0_a = self.galaxy4.best_param_dict.get("center_a", None)
        self.sigma_a = self.galaxy4.best_param_dict.get("sigma_a", None)
        self.x0_b = self.galaxy4.best_param_dict.get("center_b", None)
        self.sigma_b = self.galaxy4.best_param_dict.get("sigma_b", None)
        self.x0_b2 = self.galaxy4.best_param_dict.get("center_b2", None)
        self.sigma_b2 = self.galaxy4.best_param_dict.get("sigma_b2", None)

        # create a dict for x0 and sigma respectively
        self.x0_dict = {'x0_e': self.x0_e, 'x0_b': self.x0_b, 'x0_b2': self.x0_b2, 'x0_a': self.x0_a}
        self.sigma_dict = {'sigma_e': self.sigma_e, 'sigma_b': self.sigma_b, 'sigma_b2': self.x0_b2, 'sigma_a': self.sigma_a}

        # get the dict that contains the best-fitting amplitudes for all lines
        self.amps_dict = self.galaxy4.best_amps

        # initialize the dicts for residuals, chi-square, and best-fitting models
        self.residual_dict = dict()
        self.redchi2_dict = dict()
        # model for narrow and broad emission components
        if self.broad_wings_lines:
            self.best_model_n_dict = dict()
            self.best_model_b_dict = dict()
            if self.triple_gauss_lines:
                self.best_model_b2_dict = dict()
        # model for emission and absorption components
        if self.absorption_lines:
            self.best_model_em_dict = dict()
            self.best_model_ab_dict = dict()

        # iterate through each line and return their residuals and best-fitting models
        for line, func in self.lines_dict.items():
            if line in self.selected_lines:
                if '&' in line:  # Special case for doublet that should be fitted together
                    multilet_lines = split_multilet_line(line)
                    amps = [self.amps_dict[key] for key in line.split(' ')[1].split('&')]
                    # single emission component
                    if all((l not in self.broad_wings_lines) for l in multilet_lines):
                        params_line = [self.x0_e, self.sigma_e] + amps
                        self.residual_dict[line] = residual_2p_v_c_doublet(params_line, self.velocity_dict[line][0], self.velocity_dict[line][1], 
                                                                           self.flux_v_dict[line], self.err_v_dict[line])
                    # multi emission component
                    else:
                        for num_ii, l in enumerate(multilet_lines):
                            # Default single emission component
                            value = 1
                            # Check for double and triple emission components
                            if l in self.broad_wings_lines:
                                value = 2 if l not in self.triple_gauss_lines else 3
                            # Assign the determined value based on num_ii
                            if num_ii == 0:
                                num_comp_first = value
                            elif num_ii == 1:
                                num_comp_second = value
                            elif num_ii == 2:
                                num_comp_third = value
                        
                        # define the base of params 
                        params_line = [self.x0_e, self.sigma_e] + amps + [self.x0_b, self.sigma_b]

                        # Double line profiles
                        if len(multilet_lines) == 2:
                            # line 1
                            broad_amp_1 = self.galaxy4.get_broad_amp(self.amps_dict, num_comp_first, multilet_lines[0])
                            self.best_model_n_dict[multilet_lines[0]] = gaussian_1p_v(self.velocity_dict[line][0], self.x0_e, self.sigma_e, amps[0])
                            if broad_amp_1:
                                self.best_model_b_dict[multilet_lines[0]] = gaussian_1p_v(self.velocity_dict[line][0], self.x0_b, self.sigma_b, broad_amp_1[0])
                            # line 2
                            broad_amp_2 = self.galaxy4.get_broad_amp(self.amps_dict, num_comp_second, multilet_lines[1])
                            self.best_model_n_dict[multilet_lines[1]] = gaussian_1p_v(self.velocity_dict[line][1], self.x0_e, self.sigma_e, amps[1])
                            if broad_amp_2:
                                self.best_model_b_dict[multilet_lines[1]] = gaussian_1p_v(self.velocity_dict[line][1], self.x0_b, self.sigma_b, broad_amp_2[0])
                            params_line += broad_amp_1 + broad_amp_2
                            # check whether they have the third emission components
                            if num_comp_first == 3 or num_comp_second == 3:
                                params_line += [self.x0_b2, self.sigma_b2] 
                                # line 1
                                if num_comp_first == 3:
                                    broad_amp2_1 = self.galaxy4.get_broad_amp(self.amps_dict, num_comp_first, multilet_lines[0], "2")
                                    params_line += broad_amp2_1
                                    if broad_amp2_1:
                                        self.best_model_b2_dict[multilet_lines[0]] = gaussian_1p_v(self.velocity_dict[line][0], self.x0_b2, self.sigma_b2, broad_amp2_1[0])
                                # line 2
                                if num_comp_second == 3:
                                    broad_amp2_2 = self.galaxy4.get_broad_amp(self.amps_dict, num_comp_second, multilet_lines[1], "2")
                                    params_line += broad_amp2_2
                                    if broad_amp2_2:
                                        self.best_model_b2_dict[multilet_lines[1]] = gaussian_1p_v(self.velocity_dict[line][1], self.x0_b2, self.sigma_b2, broad_amp2_2[0])
                            # append the line residual to the residual dict
                            self.residual_dict[line] = residual_2p_v_c_doublet(params_line, self.velocity_dict[line][0], self.velocity_dict[line][1], 
                                                                               self.flux_v_dict[line], self.err_v_dict[line], 
                                                                               num_comp_first=num_comp_first, num_comp_second=num_comp_second)

                        # Triple line profiles
                        if len(multilet_lines) == 3:
                            # line 1
                            broad_amp_1 = self.galaxy4.get_broad_amp(self.amps_dict, num_comp_first, multilet_lines[0])
                            self.best_model_n_dict[multilet_lines[0]] = gaussian_1p_v(self.velocity_dict[line][0], self.x0_e, self.sigma_e, amps[0])
                            if broad_amp_1:
                                self.best_model_b_dict[multilet_lines[0]] = gaussian_1p_v(self.velocity_dict[line][0], self.x0_b, self.sigma_b, broad_amp_1[0])
                            # line 2
                            broad_amp_2 = self.galaxy4.get_broad_amp(self.amps_dict, num_comp_second, multilet_lines[1])
                            self.best_model_n_dict[multilet_lines[1]] = gaussian_1p_v(self.velocity_dict[line][1], self.x0_e, self.sigma_e, amps[1])
                            if broad_amp_2:
                                self.best_model_b_dict[multilet_lines[1]] = gaussian_1p_v(self.velocity_dict[line][1], self.x0_b, self.sigma_b, broad_amp_2[0])
                            # line 3
                            broad_amp_3 = self.galaxy4.get_broad_amp(self.amps_dict, num_comp_third, multilet_lines[2])
                            self.best_model_n_dict[multilet_lines[2]] = gaussian_1p_v(self.velocity_dict[line][2], self.x0_e, self.sigma_e, amps[2])
                            if broad_amp_3:
                                self.best_model_b_dict[multilet_lines[2]] = gaussian_1p_v(self.velocity_dict[line][2], self.x0_b, self.sigma_b, broad_amp_3[0])
                            params_line += broad_amp_1 + broad_amp_2 + broad_amp_3
                            # check whether they have the third emission components
                            if any(x == 3 for x in [num_comp_first, num_comp_second, num_comp_third]):
                                params_line += [self.x0_b2, self.sigma_b2] 
                                # line 1
                                if num_comp_first == 3:
                                    broad_amp2_1 = self.galaxy4.get_broad_amp(self.amps_dict, num_comp_first, multilet_lines[0], "2")
                                    params_line += broad_amp2_1 
                                    if broad_amp2_1:
                                        self.best_model_b2_dict[multilet_lines[0]] = gaussian_1p_v(self.velocity_dict[line][0], self.x0_b2, self.sigma_b2, broad_amp2_1[0])
                                # line 2
                                if num_comp_second == 3:
                                    broad_amp2_2 = self.galaxy4.get_broad_amp(self.amps_dict, num_comp_second, multilet_lines[1], "2")
                                    params_line += broad_amp2_2
                                    if broad_amp2_2:
                                        self.best_model_b2_dict[multilet_lines[1]] = gaussian_1p_v(self.velocity_dict[line][1], self.x0_b2, self.sigma_b2, broad_amp2_2[0])
                                # line 3
                                if num_comp_third == 3:
                                    broad_amp2_3 = self.galaxy4.get_broad_amp(self.amps_dict, num_comp_third, multilet_lines[2], "2")
                                    params_line += broad_amp2_3
                                    if broad_amp2_3:
                                        self.best_model_b2_dict[multilet_lines[2]] = gaussian_1p_v(self.velocity_dict[line][2], self.x0_b2, self.sigma_b2, broad_amp2_3[0])
                            # append the line residual to the residual dict
                            self.residual_dict[line] = residual_3p_v_c_triplet(params_line, self.velocity_dict[line][0], self.velocity_dict[line][1], 
                                                                               self.velocity_dict[line][2], self.flux_v_dict[line], self.err_v_dict[line], 
                                                                               num_comp_first=num_comp_first, num_comp_second=num_comp_second, num_comp_third=num_comp_third)
                else:  # General case
                    amp = [self.amps_dict[line.split(' ')[1]]]
                    # single line profile with single emission component
                    if (line not in self.absorption_lines) and (line not in self.broad_wings_lines):
                        params_line = [self.x0_e, self.sigma_e] + amp
                        self.residual_dict[line] = residual_1p_v_c(params_line, self.velocity_dict[line], self.flux_v_dict[line], self.err_v_dict[line])
                    # single line profile with multi emission components
                    if (line not in self.absorption_lines) and (line in self.broad_wings_lines):
                        # double emission components
                        if line in self.double_gauss_lines:
                            broad_amp = [self.amps_dict[f"{line.split(' ')[1]}_b"]]
                            params_line = [self.x0_e, self.x0_b, self.sigma_e, self.sigma_b] + amp + broad_amp
                            self.residual_dict[line] = residual_2p_v_c(params_line, self.velocity_dict[line], self.flux_v_dict[line], self.err_v_dict[line])
                            self.best_model_n_dict[line] = gaussian_1p_v(self.velocity_dict[line], self.x0_e, self.sigma_e, amp[0])
                            self.best_model_b_dict[line] = gaussian_1p_v(self.velocity_dict[line], self.x0_b, self.sigma_b, broad_amp[0])
                        # triple emission components
                        if line in self.triple_gauss_lines:
                            broad_amp = [self.amps_dict[f"{line.split(' ')[1]}_b"], self.amps_dict[f"{line.split(' ')[1]}_b2"]]
                            params_line = [self.x0_e, self.x0_b, self.x0_b2, self.sigma_e, self.sigma_b, self.sigma_b2] + amp + broad_amp
                            self.residual_dict[line] = residual_3p_v_c(params_line, self.velocity_dict[line], self.flux_v_dict[line], self.err_v_dict[line])
                            self.best_model_n_dict[line] = gaussian_1p_v(self.velocity_dict[line], self.x0_e, self.sigma_e, amp[0])
                            self.best_model_b_dict[line] = gaussian_1p_v(self.velocity_dict[line], self.x0_b, self.sigma_b, broad_amp[0])
                            self.best_model_b2_dict[line] = gaussian_1p_v(self.velocity_dict[line], self.x0_b2, self.sigma_b2, broad_amp[1])
                    # single line profile with emission+absorption components
                    if (line in self.absorption_lines) and (line not in self.broad_wings_lines):
                        abs_amp = [self.amps_dict[f"{line.split(' ')[1]}_abs"]]
                        params_line = [self.x0_e, self.x0_a, self.sigma_e, self.sigma_a] + amp + abs_amp
                        self.residual_dict[line] = residual_2p_gl_v_c(params_line, self.velocity_dict[line], self.flux_v_dict[line], self.err_v_dict[line])
                        self.best_model_em_dict[line] = gaussian_1p_v(self.velocity_dict[line], self.x0_e, self.sigma_e, amp[0])   
                        self.best_model_ab_dict[line] = lorentzian_1p_v(self.velocity_dict[line], self.x0_a, self.sigma_a, abs_amp[0])
                # append the line chi2 to the chi2 dict
                dof = self.galaxy4.best_res.nfree # degree of freedom = num_of_data - num_of_params
                self.redchi2_dict[line] = np.sum(self.residual_dict[line]**2) / dof

        # initialize the dict that contains best-fitting params/fluxes/errors in wavelength space
        self.sigma_w_dict = dict()
        self.lambda_w_dict = dict()
        self.flux_dict = dict()
        if get_error:
            self.flux_err_dict = dict()
        if self.broad_wings_lines:
            self.flux_b_dict = dict()
            self.sigma_b_w_dict = dict()
            self.lambda_b_w_dict = dict()
            if get_error:
                self.flux_b_err_dict = dict()
            if self.triple_gauss_lines:
                self.flux_b2_dict = dict()
                self.sigma_b2_w_dict = dict()
                self.lambda_b2_w_dict = dict()
                if get_error:
                    self.flux_b2_err_dict = dict()
        if self.absorption_lines:
            self.flux_abs_dict = dict()
            self.sigma_abs_w_dict = dict()
            self.lambda_abs_w_dict = dict()
            if get_error:
                self.flux_abs_err_dict = dict()

        for line, func in self.lines_dict.items():
            if line in self.selected_lines:
                if '&' in line:  # Special case for multilet that should be fitted together
                    multilet_lines = split_multilet_line(line)
                    for i, l in enumerate(multilet_lines):
                        wave_len = len(self.wave_dict[line])
                        self.sigma_w_dict[l] = self.sigma_e * self.wave_dict[line][i + int(wave_len / 2)] / self.c
                        self.lambda_w_dict[l] = self.wave_dict[line][i + int(wave_len / 2)] * (1. + (self.x0_e / self.c))
                        self.flux_dict[l] = np.abs(self.amps_dict[l.split(' ')[1]]*self.sigma_e*np.sqrt(2*np.pi))
                        # multi emission component
                        if l in self.broad_wings_lines:
                            broad_amp = self.amps_dict[f"{l.split(' ')[1]}_b"]
                            self.flux_b_dict[l] = np.abs(broad_amp*self.sigma_b*np.sqrt(2*np.pi)) 
                            self.sigma_b_w_dict[l] = self.sigma_b * self.wave_dict[line][i + int(wave_len / 2)] / self.c
                            self.lambda_b_w_dict[l] = self.wave_dict[line][i + int(wave_len / 2)] * (1. + (self.x0_b / self.c))
                            if l in self.triple_gauss_lines:
                                broad_amp2 = self.amps_dict[f"{l.split(' ')[1]}_b2"]
                                self.flux_b2_dict[l] = np.abs(broad_amp2*self.sigma_b2*np.sqrt(2*np.pi)) 
                                self.sigma_b2_w_dict[l] = self.sigma_b2 * self.wave_dict[line][i + int(wave_len / 2)] / self.c
                                self.lambda_b2_w_dict[l] = self.wave_dict[line][i + int(wave_len / 2)] * (1. + (self.x0_b2 / self.c))

                else: # General case
                    amp = self.amps_dict[line.split(' ')[1]]
                    self.sigma_w_dict[line] = self.sigma_e * self.wave_dict[line][1] / self.c
                    self.lambda_w_dict[line] = self.wave_dict[line][1] * (1. + (self.x0_e / self.c))
                    self.flux_dict[line] = np.abs(amp*self.sigma_e*np.sqrt(2*np.pi))

                    if (line not in self.absorption_lines) and (line in self.broad_wings_lines):
                        broad_amp = self.amps_dict[f"{line.split(' ')[1]}_b"]
                        self.flux_b_dict[line] = np.abs(broad_amp*self.sigma_b*np.sqrt(2*np.pi)) 
                        self.sigma_b_w_dict[line] = self.sigma_b * self.wave_dict[line][1] / self.c
                        self.lambda_b_w_dict[line] = self.wave_dict[line][1] * (1. + (self.x0_b / self.c))
                        if line in self.triple_gauss_lines:
                            broad_amp2 = self.amps_dict[f"{line.split(' ')[1]}_b2"]
                            self.flux_b2_dict[line] = np.abs(broad_amp2*self.sigma_b2*np.sqrt(2*np.pi)) 
                            self.sigma_b2_w_dict[line] = self.sigma_b2 * self.wave_dict[line][1] / self.c
                            self.lambda_b2_w_dict[line] = self.wave_dict[line][1] * (1. + (self.x0_b2 / self.c))
                    if (line in self.absorption_lines) and (line not in self.broad_wings_lines):
                        abs_amp = self.amps_dict[f"{line.split(' ')[1]}_abs"]
                        self.flux_abs_dict[line] = np.abs(abs_amp*self.sigma_a*np.sqrt(2*np.pi)) 
                        self.sigma_abs_w_dict[line] = self.sigma_a * self.wave_dict[line][1] / self.c
                        self.lambda_abs_w_dict[line] = self.wave_dict[line][1] * (1. + (self.x0_a / self.c))

        # return the error each raw flux above
        if get_error:
            # error of the velocity centers and widths
            self.x0_e_err = self.galaxy4.best_res.params["center_e"].stderr
            self.sigma_e_err = self.galaxy4.best_res.params["sigma_e"].stderr
            self.x0_dict['x0_e_err'] = self.x0_e_err
            self.sigma_dict['sigma_e_err'] = self.sigma_e_err
            if self.broad_wings_lines:
                self.x0_b_err = self.galaxy4.best_res.params["center_b"].stderr
                self.sigma_b_err = self.galaxy4.best_res.params["sigma_b"].stderr
                self.x0_dict['x0_b_err'] = self.x0_b_err
                self.sigma_dict['sigma_b_err'] = self.sigma_b_err
                if self.triple_gauss_lines:
                    self.x0_b2_err = self.galaxy4.best_res.params["center_b2"].stderr
                    self.sigma_b2_err = self.galaxy4.best_res.params["sigma_b2"].stderr
                    self.x0_dict['x0_b2_err'] = self.x0_b2_err
                    self.sigma_dict['sigma_b2_err'] = self.sigma_b2_err
            if self.absorption_lines:
                self.x0_a_err = self.galaxy4.best_res.params["center_a"].stderr
                self.sigma_a_err = self.galaxy4.best_res.params["sigma_a"].stderr
                self.x0_dict['x0_a_err'] = self.x0_a_err
                self.sigma_dict['sigma_a_err'] = self.sigma_a_err
            # error of each amplitude
            for line, func in self.lines_dict.items():
                if line in self.selected_lines:
                    if '&' in line:  # Special case for doublet that should be fitted together
                        multilet_lines = split_multilet_line(line)
                        for i, l in enumerate(multilet_lines):
                            amp = self.amps_dict[l.split(' ')[1]]
                            amp_err = self.galaxy4.best_res.params[f"amp_{l.split(' ')[1]}"].stderr
                            self.amps_dict[f"{l.split(' ')[1]}_err"] = amp_err
                            self.flux_err_dict[l] = np.sqrt((np.sqrt(2*np.pi)*amp*self.sigma_e_err)**2 + (np.sqrt(2*np.pi)*self.sigma_e*amp_err)**2)
                            # multi emission component
                            if l in self.broad_wings_lines:
                                broad_amp = self.amps_dict[f"{l.split(' ')[1]}_b"]
                                broad_amp_err = self.galaxy4.best_res.params[f"amp_{l.split(' ')[1]}_b"].stderr
                                self.amps_dict[f"{l.split(' ')[1]}_b_err"] = broad_amp_err
                                self.flux_b_err_dict[l] = np.sqrt((np.sqrt(2*np.pi)*broad_amp*self.sigma_b_err)**2 + (np.sqrt(2*np.pi)*self.sigma_b*broad_amp_err)**2)
                                if l in self.triple_gauss_lines:
                                    broad_amp2 = self.amps_dict[f"{l.split(' ')[1]}_b2"]
                                    broad_amp2_err = self.galaxy4.best_res.params[f"amp_{l.split(' ')[1]}_b2"].stderr
                                    self.amps_dict[f"{l.split(' ')[1]}_b2_err"] = broad_amp2_err
                                    self.flux_b2_err_dict[l] = np.sqrt((np.sqrt(2*np.pi)*broad_amp2*self.sigma_b2_err)**2 + (np.sqrt(2*np.pi)*self.sigma_b2*broad_amp2_err)**2)
                    else: # General case
                        # flux error of the narrow emission component
                        amp = self.amps_dict[line.split(' ')[1]]
                        amp_err = self.galaxy4.best_res.params[f"amp_{line.split(' ')[1]}"].stderr
                        self.amps_dict[f"{line.split(' ')[1]}_err"] = amp_err
                        self.flux_err_dict[line] = np.sqrt((np.sqrt(2*np.pi)*amp*self.sigma_e_err)**2 + (np.sqrt(2*np.pi)*self.sigma_e*amp_err)**2)
                        # flux error of the broad emission component
                        if (line not in self.absorption_lines) and (line in self.broad_wings_lines):
                            broad_amp = self.amps_dict[f"{line.split(' ')[1]}_b"]
                            broad_amp_err = self.galaxy4.best_res.params[f"amp_{line.split(' ')[1]}_b"].stderr
                            self.amps_dict[f"{line.split(' ')[1]}_b_err"] = broad_amp_err
                            self.flux_b_err_dict[line] = np.sqrt((np.sqrt(2*np.pi)*broad_amp*self.sigma_b_err)**2 + (np.sqrt(2*np.pi)*self.sigma_b*broad_amp_err)**2)
                            if line in self.triple_gauss_lines:
                                broad_amp2 = self.amps_dict[f"{line.split(' ')[1]}_b2"]
                                broad_amp2_err = self.galaxy4.best_res.params[f"amp_{line.split(' ')[1]}_b2"].stderr
                                self.amps_dict[f"{line.split(' ')[1]}_b2_err"] = broad_amp2_err
                                self.flux_b2_err_dict[line] = np.sqrt((np.sqrt(2*np.pi)*broad_amp2*self.sigma_b2_err)**2 + (np.sqrt(2*np.pi)*self.sigma_b2*broad_amp2_err)**2)
                        # flux error of the absorption component
                        if (line in self.absorption_lines) and (line not in self.broad_wings_lines):
                            abs_amp = self.amps_dict[f"{line.split(' ')[1]}_abs"]
                            abs_amp_err = self.galaxy4.best_res.params[f"amp_{line.split(' ')[1]}_abs"].stderr
                            self.amps_dict[f"{line.split(' ')[1]}_abs_err"] = abs_amp_err
                            self.flux_abs_err_dict[line] = np.sqrt((np.sqrt(2*np.pi)*abs_amp*self.sigma_a_err)**2 + (np.sqrt(2*np.pi)*self.sigma_a*abs_amp_err)**2)

        # define the subfolder name for saving the results
        if save_ew_table or save_flux_table or save_par_table:
            # Create the main window (though it won't be shown)
            root = tk.Tk()
            root.withdraw()  # Hides the main window
            # Prompt the user for input with a default value
            self.folder_name = askstring("Input", "Please enter a subfolder name for your saved results:", initialvalue="None")

        # return the ew of each balmer line in wavelength space
        if get_ew:
            self.find_ew(save_ew_table = save_ew_table)

        # whether to save the parameter table 
        if save_par_table:
            self.save_par_pd_table()

        # whether to return the flux (extinction corrected or not)
        if get_flux and get_corr:
            corr = extinction()
            corr.E_BV = self.E_BV
            for line, func in self.lines_dict.items():
                if line in self.selected_lines:
                    if '&' in line:  # Special case for doublet that should be fitted together
                        multilet_lines = split_multilet_line(line)
                        gC_line1 = corr.find_fmcorr_flux(self.flux_dict[multilet_lines[0]],self.lambda_w_dict[multilet_lines[0]])[0]
                        rC_line1 = corr.find_gordan_flux(gC_line1,self.lambda_w_dict[multilet_lines[0]],self.A_V)
                        gC_line2 = corr.find_fmcorr_flux(self.flux_dict[multilet_lines[1]],self.lambda_w_dict[multilet_lines[1]])[0]
                        rC_line2 = corr.find_gordan_flux(gC_line2,self.lambda_w_dict[multilet_lines[1]],self.A_V)
                        self.flux_dict[line] = np.array([rC_line1, rC_line2])
                        if len(multilet_lines) == 3:
                            gC_line3 = corr.find_fmcorr_flux(self.flux_dict[multilet_lines[2]],self.lambda_w_dict[multilet_lines[2]])[0]
                            rC_line3 = corr.find_gordan_flux(gC_line3,self.lambda_w_dict[multilet_lines[2]],self.A_V)
                            self.flux_dict[line] = np.append(self.flux_dict[line], rC_line3)
                        if get_error:
                            err1 = self.flux_err_dict[multilet_lines[0]] * rC_line1 / self.flux_dict[multilet_lines[0]]
                            err2 = self.flux_err_dict[multilet_lines[1]] * rC_line2 / self.flux_dict[multilet_lines[1]]
                            self.flux_err_dict[line] = np.array([err1, err2])
                            if len(multilet_lines) == 3:
                                err3 = self.flux_err_dict[multilet_lines[2]] * rC_line3 / self.flux_dict[multilet_lines[2]]
                                self.flux_err_dict[line] = np.append(self.flux_err_dict[line], err3)
                    else: # General case for single line profile
                        gC_line = corr.find_fmcorr_flux(self.flux_dict[line],self.lambda_w_dict[line])[0]
                        rC_line = corr.find_gordan_flux(gC_line,self.lambda_w_dict[line],self.A_V)
                        self.flux_dict[line] = rC_line
                        if get_error:
                            err = self.flux_err_dict * rC_line / self.flux_dict[line]
                            self.flux_err_dict[line] = err
            # whether to save the flux table
            if save_flux_table:
                self.save_flux_pd_table()
            if get_error:
                return (self.flux_dict, self.flux_err_dict)
            if not get_error:
                return self.flux_dict
        if (get_flux) and (not get_corr):
            # whether to save the flux table
            if save_flux_table:
                self.save_flux_pd_table()
            if get_error:
                return (self.flux_dict, self.flux_err_dict)
            if not get_error:
                return self.flux_dict

        return self.galaxy4.best_params


    def fitting_input_params(self, wave, spec, espec, n_iteration = 1000):
        """
        Construct the arrays for initial values and the parameter ranges for fitting multiple emission and/or absorption lines in a galaxy spectrum (in velocity space). 
        This function is for fitting every line simultaneously.

        Parameters
        ----------
        wave : array_like
            Wavelength array.
        spec : array_like
            Flux array.
        espec : array_like
            Error array.
        n_iteration : int, optional
            Number of iterations for the MCMC algorithm. Default is 1000.
        Returns
        -------
        best_model : ndarray
            Best-fit model array for each line.
        best_chi2 : float
            Chi-square value of the best-fit model.
        """

        # get rid of nan values in the input data to avoid errors in the LMFIT line-fitting algorithm
        # TODO: Are there other ways to solve this issue? Like masking those nan-value pixels and set it to zero
        # nan_index = np.isnan(spec) | np.isnan(espec) # nan-value index
        # # copy of wave, spec, and espec
        # wave_c = np.copy(wave[np.logical_not(nan_index)])
        # spec_c = np.copy(spec[np.logical_not(nan_index)])
        # espec_c = np.copy(espec[np.logical_not(nan_index)])

        # step 1: obtain the velocity array, flux array (in velocity space), and error array (in velocity space)
        # initialize the velocity, flux, and flux_err dictionaries
        self.velocity_dict = dict()
        self.flux_v_dict = dict()
        self.err_v_dict = dict()

        # Iterate through each selected line and its corresponding function
        for line, func in self.lines_dict.items():
            if line in self.selected_lines:
                if '&' in line:  # Special case for multilet
                    multilet_lines = split_multilet_line(line)
                    # doublet
                    if len(multilet_lines) == 2:
                        v_arr, v_arr_2, flux_v_arr, err_v_arr = func(wave, spec, espec)
                        self.velocity_dict[line] = np.array([v_arr, v_arr_2])
                    # triplet
                    if len(multilet_lines) == 3:
                        v_arr, v_arr_2, v_arr_3, flux_v_arr, err_v_arr = func(wave, spec, espec)
                        self.velocity_dict[line] = np.array([v_arr, v_arr_2, v_arr_3])
                    self.flux_v_dict[line] = flux_v_arr
                    self.err_v_dict[line] = err_v_arr
                else:  # General case
                    velocity, flux, error = func(wave, spec, espec)
                    self.velocity_dict[line] = velocity
                    self.flux_v_dict[line] = flux
                    self.err_v_dict[line] = error
                    
        # Initialize the initial guess and the parameter range dictionaries
        initial_guess_dict = dict()
        param_range_dict = dict()

        # append the initial guess and the param range for the velocity center and width
        initial_guess_dict['v_e'] = np.array([self.initial_guess_dict['center_e'], self.initial_guess_dict['sigma_e']])
        param_range_dict['v_e'] = np.array([self.param_range_dict['center_e'], self.param_range_dict['sigma_e']])
        if len(self.absorption_lines) != 0:
            initial_guess_dict['v_a'] = np.array([self.initial_guess_dict['center_a'], self.initial_guess_dict['sigma_a']])
            param_range_dict['v_a'] = np.array([self.param_range_dict['center_a'], self.param_range_dict['sigma_a']])
        if len(self.broad_wings_lines) != 0:
            # two velocity emission components
            initial_guess_dict['v_b'] = np.array([self.initial_guess_dict['center_b'], self.initial_guess_dict['sigma_b']])
            param_range_dict['v_b'] = np.array([self.param_range_dict['center_b'], self.param_range_dict['sigma_b']])
            # three velocity emission components
            if len(self.triple_gauss_lines) != 0:
                initial_guess_dict['v_b2'] = np.array([self.initial_guess_dict['center_b2'], self.initial_guess_dict['sigma_b2']])
                param_range_dict['v_b2'] = np.array([self.param_range_dict['center_b2'], self.param_range_dict['sigma_b2']])

        # Iterate through each selected line and its corresponding function to append the initial guess and param range for the line profile amplitude
        for line, func in self.lines_dict.items():
            if line in self.selected_lines:
                flux_line_max = np.max(self.flux_v_dict[line]) # maximum of the line profile
                # Handle multilet lines
                if '&' in line:
                    multilet_lines = split_multilet_line(line)
                    for subline in multilet_lines:
                        line_amp_base = f"amp_{subline.split(' ')[1]}" # base of the line amplitude name
                        case_amp_keys = [line_amp_base]

                        # Multi-emission component case
                        if subline in self.broad_wings_lines:
                            case_amp_keys.append(f"{line_amp_base}_b")
                            # Three velocity emission components
                            if subline in self.triple_gauss_lines:
                                case_amp_keys.append(f"{line_amp_base}_b2")

                        # Absorption case
                        if subline in self.absorption_lines:
                            case_amp_keys.append(f"{line_amp_base}_abs")

                        # Populate dicts for current line
                        initial_guess_dict[subline] = flux_line_max * np.array([self.initial_guess_dict[key] for key in case_amp_keys])
                        param_range_dict[subline] = flux_line_max * np.array([self.param_range_dict[key] for key in case_amp_keys])
                    continue

                # Single line case
                line_amp_base = f"amp_{line.split(' ')[1]}" # base of the line amplitude name
                case_amp_keys = [line_amp_base]

                # Multi-emission component case
                if line in self.broad_wings_lines:
                    case_amp_keys.append(f"{line_amp_base}_b")
                    # Three velocity emission components
                    if line in self.triple_gauss_lines:
                        case_amp_keys.append(f"{line_amp_base}_b2")

                # Absorption case
                if line in self.absorption_lines:
                    case_amp_keys.append(f"{line_amp_base}_abs")

                # Populate dicts for current line
                initial_guess_dict[line] = flux_line_max * np.array([self.initial_guess_dict[key] for key in case_amp_keys])
                param_range_dict[line] = flux_line_max * np.array([self.param_range_dict[key] for key in case_amp_keys])

        input_arr = (self.velocity_dict, self.flux_v_dict, self.err_v_dict, initial_guess_dict, param_range_dict, self.amps_ratio_dict, self.absorption_lines, 
                     self.broad_wings_lines, self.double_gauss_lines, self.triple_gauss_lines, self.double_gauss_broad, self.triple_gauss_broad, self.fitting_method)
        self.galaxy4 = line_fitting_model(seed = self.seed)
        best_model, best_chi2 = self.galaxy4.fitting_all_lines(input_arr, n_iteration = n_iteration)
        return best_model, best_chi2

    def save_par_pd_table(self):
        # Define col names
        col_names = ["velocity center", "velocity width", "line amplitude"]
        
        # Create empty dataframes list
        dfs = []

        # Create separate dataframes for each parameter
        for i, par_dict in enumerate([self.x0_dict, self.sigma_dict, self.amps_dict]):
            row = {}

            if i in [0, 1]:  # x0 and sigma
                key = 'x0' if i == 0 else 'sigma'
                for component in ['e', 'b', 'b2', 'a']:
                    row.update({
                        f"{key}_{component}": getattr(self, f"{key}_{component}", np.nan),
                        f"{key}_{component}_err": getattr(self, f"{key}_{component}_err", np.nan),
                    })
            else:  # amplitude
                row = {
                    f"amp_{amp}": self.amps_dict.get(amp, np.nan) for amp in self.amps_dict.keys()
                }

            # Append the row DataFrame into the dfs list
            dfs.append(pd.DataFrame(row, index=[col_names[i]]).T)

        # Concatenate all the DataFrames in dfs
        self.par_df = pd.concat(dfs)
            
        # Define the parent directory for the flux table
        if self.folder_name != 'None':
            directory = f"parameter_tables/{self.folder_name}/"
        if self.folder_name == 'None':
            directory = 'parameter_tables/'
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the whole DataFrame to a CSV file in the directory
        self.par_df.to_csv(os.path.join(directory, self.fits_name + '_parameters.csv'))

    def save_flux_pd_table(self):
        '''
        Saves a Pandas DataFrame containing the fluxes of each line to a CSV file.

        The DataFrame contains columns for each attribute of the object (flux_e, flux_e_err, etc.). If an attribute does not exist, 
        it is replaced with an empty dictionary in the DataFrame. The DataFrame is then saved to a CSV file in the 'flux_tables' directory.

        This method does not return anything; it simply saves the DataFrame to a file.
        '''
        self.flux_df = pd.DataFrame({
            'flux_e': self.flux_dict if hasattr(self, 'flux_dict') else {},
            'flux_e_err': self.flux_err_dict if hasattr(self, 'flux_err_dict') else {},
            'flux_b': self.flux_b_dict if hasattr(self, 'flux_b_dict') else {},
            'flux_b_err': self.flux_b_err_dict if hasattr(self, 'flux_b_err_dict') else {},
            'flux_b2': self.flux_b2_dict if hasattr(self, 'flux_b2_dict') else {},
            'flux_b2_err': self.flux_b2_err_dict if hasattr(self, 'flux_b2_err_dict') else {},
            'flux_abs': self.flux_abs_dict if hasattr(self, 'flux_abs_dict') else {},
            'flux_abs_err': self.flux_abs_err_dict if hasattr(self, 'flux_abs_err_dict') else {}
        })
        print(self.flux_df)

        # Define the parent directory for the flux table
        if self.folder_name != 'None':
            directory = f"flux_tables/{self.folder_name}/"
        if self.folder_name == 'None':
            directory = 'flux_tables/'
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the whole DataFrame to a CSV file in the directory
        self.flux_df.to_csv(os.path.join(directory, self.fits_name + '_flux.csv'))

    def save_ew_pd_table(self):
        '''
        Saves a Pandas DataFrame containing the equivalent widths (EW) of each line to a CSV file.

        The DataFrame contains columns for each attribute of the object (ew_all, ew_all_err, etc.). If an attribute does not exist, 
        it is replaced with an empty dictionary in the DataFrame. The DataFrame is then saved to a CSV file in the 'ew_tables' directory.

        This method does not return anything; it simply saves the DataFrame to a file.
        '''
        self.ew_df = pd.DataFrame({
            'ew_all': self.ew_all_dict if hasattr(self, 'ew_all_dict') else {},
            'ew_all_err': self.ew_all_err_dict if hasattr(self, 'ew_all_err_dict') else {},
            'ew_e': self.ew_e_dict if hasattr(self, 'ew_e_dict') else {},
            'ew_e_err': self.ew_e_err_dict if hasattr(self, 'ew_e_err_dict') else {},
            'ew_b': self.ew_b_dict if hasattr(self, 'ew_b_dict') else {},
            'ew_b_err': self.ew_b_err_dict if hasattr(self, 'ew_b_err_dict') else {},
            'ew_b2': self.ew_b2_dict if hasattr(self, 'ew_b2_dict') else {},
            'ew_b2_err': self.ew_b2_err_dict if hasattr(self, 'ew_b2_err_dict') else {},
            'ew_abs': self.ew_abs_dict if hasattr(self, 'ew_abs_dict') else {},
            'ew_abs_err': self.ew_abs_err_dict if hasattr(self, 'ew_abs_err_dict') else {}
        })
        print(self.ew_df)
        # define the parent directory for the ew table
        if self.folder_name != 'None':
            directory = f"ew_tables/{self.folder_name}/"
        if self.folder_name == 'None':
            directory = 'ew_tables/'
        # Create parent directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the whole DataFrame to a CSV file in the parent directory
        self.ew_df.to_csv(os.path.join(directory, self.fits_name + '_ew.csv'))


    def find_ew(self, save_ew_table = False):
        # assume we have already got the continuum level for each balmer line; 
        # we also have the best-fitting model for each balmer line in the velocity space 

        '''
        This function find_ew_balmer is used to find the equivalent width (EW) of Balmer lines in the wavelength space.

        The function takes a boolean absorption flag as input, which is used to determine whether to calculate the EW for absorption troughs or not. 
        By default, absorption is set to False.

        '''

        # initialize the ew dict for all intended lines
        self.ew_all_dict = dict()
        self.ew_e_dict = dict()
        self.ew_all_err_dict = dict()
        self.ew_e_err_dict = dict()
        if self.broad_wings_lines:
            self.ew_b_dict = dict()
            self.ew_b_err_dict = dict()
            if self.triple_gauss_lines:
                self.ew_b2_dict = dict()
                self.ew_b2_err_dict = dict()
        if self.absorption_lines:
            self.ew_abs_dict = dict()
            self.ew_abs_err_dict = dict()

        # convert the best-fitting model from the velocity space to the wavelength space
        for line, func in self.lines_dict.items():
            if line in self.selected_lines:
                # doublet that needs to be fitted together
                if '&' in line:  # Special case for doublet that should be fitted together
                    multilet_lines = split_multilet_line(line)
                    wave_len = len(self.wave_dict[line])
                    for ii, l in enumerate(multilet_lines):
                        # derive the ew for the combined line profile
                        self.model_w = self.model_dict[l] * self.c / self.wave_dict[line][ii + int(wave_len / 2)]
                        wave_line = ((self.velocity_dict[line][ii] / self.c) + 1.) * self.wave_dict[line][ii + int(wave_len / 2)] 
                        flux_all = (-self.model_w) / self.cont_line_dict[line][0]
                        ew_all = calc_ew(self.model_w, wave_line, self.cont_line_dict[line][0])
                        self.ew_all_dict[l] = ew_all
                        ew_all_err = calc_ew_err(self.model_w, self.sigma_w_dict[l], self.cont_line_dict[line][0])
                        self.ew_all_err_dict[l] = ew_all_err

                        # derive the ew for the narrow and broad components of the line profile
                        if (l in self.broad_wings_lines):
                            self.best_model_n_w = self.best_model_n_dict[l] * self.c / self.wave_dict[line][ii + int(wave_len / 2)]
                            self.best_model_b_w = self.best_model_b_dict[l] * self.c / self.wave_dict[line][ii + int(wave_len / 2)]
                            ew_b = calc_ew(self.best_model_b_w, wave_line, self.cont_line_dict[line][0])
                            self.ew_b_dict[l] = ew_b
                            ew_b_err = calc_ew_err(self.best_model_b_w, self.sigma_b_w_dict[l], self.cont_line_dict[line][0])
                            self.ew_b_err_dict[l] = ew_b_err
                            ew_e = calc_ew(self.best_model_n_w, wave_line, self.cont_line_dict[line][0])
                            self.ew_e_dict[l] = ew_e
                            ew_e_err = calc_ew_err(self.best_model_n_w, self.sigma_w_dict[l], self.cont_line_dict[line][0])
                            self.ew_e_err_dict[l] = ew_e_err
                            # if there is a third emission component
                            if line in self.triple_gauss_lines:
                                self.best_model_b2_w = self.best_model_b2_dict[l] * self.c / self.wave_dict[line][ii + int(wave_len / 2)]
                                ew_b2 = calc_ew(self.best_model_b2_w, wave_line, self.cont_line_dict[line][0])
                                self.ew_b2_dict[l] = ew_b2
                                ew_b2_err = calc_ew_err(self.best_model_b2_w, self.sigma_b2_w_dict[l], self.cont_line_dict[line][0])
                                self.ew_b2_err_dict[l] = ew_b2_err
                # single line profile
                else: # General case
                    # derive the ew for the combined line profile
                    self.model_w = self.model_dict[line] * self.c / self.wave_dict[line][-1]
                    wave_line = ((self.velocity_dict[line] / self.c) + 1.) * self.wave_dict[line][-1] 
                    flux_all = (-self.model_w) / self.cont_line_dict[line][0]
                    ew_all = calc_ew(self.model_w, wave_line, self.cont_line_dict[line][0])
                    self.ew_all_dict[line] = ew_all
                    ew_all_err = calc_ew_err(self.model_w, self.sigma_w_dict[line], self.cont_line_dict[line][0])
                    self.ew_all_err_dict[line] = ew_all_err
                    # derive the ew for the absorption and emission components of the line profile
                    if (line in self.absorption_lines) and (line not in self.broad_wings_lines):
                        self.best_model_ab_w = self.best_model_ab_dict[line] * self.c / self.wave_dict[line][-1]
                        self.best_model_em_w = self.best_model_em_dict[line] * self.c / self.wave_dict[line][-1]
                        ew_a = calc_ew(self.best_model_ab_w, wave_line, self.cont_line_dict[line][0])
                        self.ew_abs_dict[line] = ew_a
                        ew_a_err = calc_ew_err(self.best_model_ab_w, self.sigma_abs_w_dict[line], self.cont_line_dict[line][0])
                        self.ew_abs_err_dict[line] = ew_a_err
                        ew_e = calc_ew(self.best_model_em_w, wave_line, self.cont_line_dict[line][0])
                        self.ew_e_dict[line] = ew_e
                        ew_e_err = calc_ew_err(self.best_model_em_w, self.sigma_w_dict[line], self.cont_line_dict[line][0])
                        self.ew_e_err_dict[line] = ew_e_err
                    # derive the ew for the narrow and broad components of the line profile
                    if (line not in self.absorption_lines) and (line in self.broad_wings_lines):
                        self.best_model_n_w = self.best_model_n_dict[line] * self.c / self.wave_dict[line][-1]
                        self.best_model_b_w = self.best_model_b_dict[line] * self.c / self.wave_dict[line][-1]
                        ew_b = calc_ew(self.best_model_b_w, wave_line, self.cont_line_dict[line][0])
                        self.ew_b_dict[line] = ew_b
                        ew_b_err = calc_ew_err(self.best_model_b_w, self.sigma_b_w_dict[line], self.cont_line_dict[line][0])
                        self.ew_b_err_dict[line] = ew_b_err
                        ew_e = calc_ew(self.best_model_n_w, wave_line, self.cont_line_dict[line][0])
                        self.ew_e_dict[line] = ew_e
                        ew_e_err = calc_ew_err(self.best_model_n_w, self.sigma_w_dict[line], self.cont_line_dict[line][0])
                        self.ew_e_err_dict[line] = ew_e_err
                        # if there is a third emission component
                        if line in self.triple_gauss_lines:
                            self.best_model_b2_w = self.best_model_b2_dict[line] * self.c / self.wave_dict[line][-1]
                            ew_b2 = calc_ew(self.best_model_b2_w, wave_line, self.cont_line_dict[line][0])
                            self.ew_b2_dict[line] = ew_b2
                            ew_b2_err = calc_ew_err(self.best_model_b2_w, self.sigma_b2_w_dict[line], self.cont_line_dict[line][0])
                            self.ew_b2_err_dict[line] = ew_b2_err
        # whether to save the ew table that shows the ew of each line or not
        if save_ew_table:
            self.save_ew_pd_table()
            # self.save_ew_directories()

    def fitting_plot(self, savefig = True):
        """
        Creates a figure with subplots displaying the spectral line profiles and their corresponding fitted models. 
        Each subplot also includes a residuals plot. The fitted model may include different components based on 
        the type of line (e.g., single line, doublet, triplet). The fitting includes narrow Gaussian, broad wings, 
        and potentially an extra broad component (triple Gaussian), as well as absorption lines. 
        The results are normalized and plotted in velocity space.

        The top row of subplots display the raw and best-fitting line profile (with all components if they exist),
        along with the error bar. The y-axis is in logarithmic scale. 

        The bottom row shows the residuals of the fit, with horizontal lines at y=0, y=-1, and y=1 for reference.

        If `savefig` is True, the plot is saved as a pdf in a sub-directory called 'plots'. The file is named 
        according to the fits_name attribute with '_fittings.pdf' appended.

        Parameters:
        savefig (bool): If True, the plot is saved as a pdf. Default is True.
        
        Returns:
        None
        """
        # determine the number of plots based on the number of selected lines
        num_plots = len(self.selected_lines)
        
        ## Plot Styling
        plt.rcParams['font.family'] = "serif"
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        matplotlib.rcParams['xtick.top'] = True
        matplotlib.rcParams['ytick.right'] = True
        matplotlib.rcParams['xtick.minor.visible'] = True
        matplotlib.rcParams['ytick.minor.visible'] = True
        matplotlib.rcParams['lines.dash_capstyle'] = "round"
        matplotlib.rcParams['lines.solid_capstyle'] = "round"
        matplotlib.rcParams['legend.handletextpad'] = 0.4
        matplotlib.rcParams['axes.linewidth'] = 0.6
        matplotlib.rcParams['ytick.major.width'] = 0.6
        matplotlib.rcParams['xtick.major.width'] = 0.6
        matplotlib.rcParams['ytick.minor.width'] = 0.45
        matplotlib.rcParams['xtick.minor.width'] = 0.45
        matplotlib.rcParams['ytick.major.size'] = 2.75
        matplotlib.rcParams['xtick.major.size'] = 2.75
        matplotlib.rcParams['ytick.minor.size'] = 1.75
        matplotlib.rcParams['xtick.minor.size'] = 1.75
        matplotlib.rcParams['legend.handlelength'] = 2

        # create the plotting figure
        fig, axes = plt.subplots(2, num_plots, gridspec_kw={'height_ratios': [2.5, 1]}, 
                                 figsize=(6.4*num_plots, 6.7))
        # eliminate the spacing between the top and bottom panels
        plt.subplots_adjust(hspace = 0)

        # If only one line is selected, convert axes to 2D array for consistency
        if num_plots == 1:
            axes = np.array(axes).reshape(2, -1)
        # Turn off x tick labels for all subplots in top row
        for ax in axes[0]:
            ax.set_xticklabels([])

        # initialize the local continuum of each line profile in velocity space
        self.cont_line_v_dict = dict()
        self.flux_plus_cont_v_dict = dict()
        self.model_plus_cont_v_dict = dict()
        # whether include broad_wings_lines
        if self.broad_wings_lines:
            self.best_model_plus_cont_n_v_dict = dict()
            self.best_model_plus_cont_b_v_dict = dict()
            if self.triple_gauss_lines:
                self.best_model_plus_cont_b2_v_dict = dict()
        # whether include absorption_lines
        if self.absorption_lines:
            self.best_model_plus_cont_em_v_dict = dict()
            self.best_model_plus_cont_ab_v_dict = dict()

        for line, func in self.lines_dict.items():
            if line in self.selected_lines:
                # doublet that needs to be fitted together
                if '&' in line:  # Special case for doublet that should be fitted together
                    multilet_lines = split_multilet_line(line)
                    self.cont_line_v_dict[line] = self.cont_line_dict[line][0] * self.lambda_w_dict[multilet_lines[0]] / self.c
                    for ii, l in enumerate(multilet_lines):
                        self.model_plus_cont_v_dict[l] = self.model_dict[l] + self.cont_line_v_dict[line]
                        if l in self.broad_wings_lines:
                            self.best_model_plus_cont_n_v_dict[l] = self.best_model_n_dict[l] + self.cont_line_v_dict[line]
                            self.best_model_plus_cont_b_v_dict[l] = self.best_model_b_dict[l] + self.cont_line_v_dict[line]
                            if l in self.triple_gauss_lines:
                                self.best_model_plus_cont_b2_v_dict[l] = self.best_model_b2_dict[l] + self.cont_line_v_dict[line]
                # single line profile
                else: # General case
                    self.cont_line_v_dict[line] = self.cont_line_dict[line][0] * self.lambda_w_dict[line] / self.c
                self.flux_plus_cont_v_dict[line] = self.flux_v_dict[line] + self.cont_line_v_dict[line]
                self.model_plus_cont_v_dict[line] = self.model_dict[line] + self.cont_line_v_dict[line]
                # whether include broad_wings_lines
                if line in self.broad_wings_lines:
                    self.best_model_plus_cont_n_v_dict[line] = self.best_model_n_dict[line] + self.cont_line_v_dict[line]
                    self.best_model_plus_cont_b_v_dict[line] = self.best_model_b_dict[line] + self.cont_line_v_dict[line]
                    if line in self.triple_gauss_lines:
                        self.best_model_plus_cont_b2_v_dict[line] = self.best_model_b2_dict[line] + self.cont_line_v_dict[line]
                # whether include absorption_lines
                if line in self.absorption_lines:
                    self.best_model_plus_cont_em_v_dict[line] = self.best_model_em_dict[line] + self.cont_line_v_dict[line]
                    self.best_model_plus_cont_ab_v_dict[line] = self.best_model_ab_dict[line] + self.cont_line_v_dict[line]

        # plot the fitting results
        for i, line in enumerate(self.selected_lines):
            if '&' in line:
                multilet_lines = split_multilet_line(line)
                line_name = '&'.join(multilet_lines[:min(len(multilet_lines), 3)])
                v_arr = self.velocity_dict[line][0] if len(multilet_lines) == 2 else self.velocity_dict[line][1]
                for l in multilet_lines:
                    if l in self.broad_wings_lines:
                        axes[0,i].plot(v_arr, self.best_model_plus_cont_n_v_dict[l] / np.max(self.flux_plus_cont_v_dict[line]), 'c--',
                                       zorder = 3, lw = 2)
                        axes[0,i].plot(v_arr, self.best_model_plus_cont_b_v_dict[l] / np.max(self.flux_plus_cont_v_dict[line]), 'b--',
                                       zorder = 3, lw = 2)
                        if l in self.triple_gauss_lines:
                            axes[0,i].plot(v_arr, self.best_model_plus_cont_b2_v_dict[l] / np.max(self.flux_plus_cont_v_dict[line]), 'g--',
                                           zorder = 3, lw = 2)
            else:
                v_arr = self.velocity_dict[line]
            if '&' not in line:
                line_name = line
            # upper panel for plotting the raw and best-fitting line profile
            axes[0,i].step(v_arr, self.flux_plus_cont_v_dict[line]/np.max(self.flux_plus_cont_v_dict[line]), 'k', where = 'mid')
            axes[0,i].fill_between(v_arr, (self.flux_plus_cont_v_dict[line]+self.err_v_dict[line]) / np.max(self.flux_plus_cont_v_dict[line]),
                                  (self.flux_plus_cont_v_dict[line]-self.err_v_dict[line]) / np.max(self.flux_plus_cont_v_dict[line]), alpha =0.5, zorder = 1,
                                   facecolor = 'black')
            axes[0,i].plot(v_arr, self.model_plus_cont_v_dict[line] / np.max(self.flux_plus_cont_v_dict[line]), 'r--', zorder = 2, lw = 2)
            if line in self.broad_wings_lines:
                axes[0,i].plot(v_arr, self.best_model_plus_cont_n_v_dict[line] / np.max(self.flux_plus_cont_v_dict[line]), 'c--',
                               zorder = 3, lw = 2)
                axes[0,i].plot(v_arr, self.best_model_plus_cont_b_v_dict[line] / np.max(self.flux_plus_cont_v_dict[line]), 'b--',
                               zorder = 3, lw = 2)
                if line in self.triple_gauss_lines:
                    axes[0,i].plot(v_arr, self.best_model_plus_cont_b2_v_dict[line] / np.max(self.flux_plus_cont_v_dict[line]), 'g--',
                                   zorder = 3, lw = 2)
            if line in self.absorption_lines:
                axes[0,i].plot(v_arr, self.best_model_plus_cont_em_v_dict[line] / np.max(self.flux_plus_cont_v_dict[line]), 'c--',
                               zorder = 2, lw = 2)
                axes[0,i].plot(v_arr, self.best_model_plus_cont_ab_v_dict[line] / np.max(self.flux_plus_cont_v_dict[line]), 'b--',
                               zorder = 2, lw = 2)
            axes[0,i].set_yscale('log')
            axes[0,i].text(0.04, 0.92, line_name + '\n' + r'$\chi^2 = $' + "{0:.2f}".format(self.redchi2_dict[line]), 
                           size = 14, transform=axes[0,i].transAxes, va="center",color="black")
            if 'H alpha' in line_name:
                axes[0,i].set_xlim(-2000, 2000)
                axes[1,i].set_xlim(-2000, 2000)
            if '[OIII]' in line_name:
                vmin, vmax = np.nanpercentile(v_arr, 10), np.nanpercentile(v_arr, 90)
                axes[0,i].set_xlim(-1500, 1500)
                axes[1,i].set_xlim(-1500, 1500)
            if 'H beta' in line_name:
                vmin, vmax = np.nanpercentile(v_arr, 10), np.nanpercentile(v_arr, 90)
                axes[0,i].set_xlim(-1000, 1000)
                axes[1,i].set_xlim(-1000, 1000)
            if '[SII]' in line_name:
                axes[0,i].set_xlim(-800, 1200)
                axes[1,i].set_xlim(-800, 1200)
            # if '&' not in line_name: 
            #     vmin, vmax = np.nanpercentile(v_arr, 15), np.nanpercentile(v_arr, 85)
            #     axes[0,i].set_xlim(-1000, 1000)
            #     axes[1,i].set_xlim(-1000, 1000) 
            axes[0,i].tick_params(axis='y', which='minor')
            axes[0,i].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            if i == 0:
                axes[0,i].set_ylabel(r'Normalized Flux',size = 22)
            # axes[0,i].legend(loc='upper left', fontsize = 13, framealpha = 0)
            # lower panel for plotting the residual
            axes[1,i].step(v_arr, self.residual_dict[line], where = 'mid')
            [axes[1,i].axhline(y=j, color="red", linestyle='--') for j in [0,-1,1]]
            axes[1,i].set_xlabel(r'Velocity $\mathrm{(km \ s^{-1})}$',size = 22)
            if i == 0:
                axes[1,i].set_ylabel(r'Residual',size = 22)
        # whether to save the figure
        if savefig:
            # define the current working directory
            current_direc = os.getcwd()
            # find whether the folder name exists or not
            try:
                self.folder_name
            except:
                # Create the main window (though it won't be shown)
                root = tk.Tk()
                root.withdraw()  # Hides the main window
                # Prompt the user for input with a default value
                self.folder_name = askstring("Input", "Please enter a subfolder name for your saved results:", initialvalue="None")
            # define the results directory based on the sub-folder name
            if self.folder_name != 'None':
                results_dir = os.path.join(current_direc, f"plots/{self.folder_name}/")
            if self.folder_name == 'None':
                results_dir = os.path.join(current_direc, f"plots/")
            # results_dir = os.path.join(current_direc, 'plots/HeII/')
            if not os.path.isdir(results_dir):
                os.mkdir(results_dir)
            # define the filename and save it in the sub-directory output_files
            filename = self.fits_name + '_fittings.pdf'
            fig.savefig(results_dir+filename, dpi=300, bbox_inches='tight')
        plt.show()


     