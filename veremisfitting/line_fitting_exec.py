import numpy as np
from astropy.io import fits
import os, sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy import stats
from scipy.stats import skew
from scipy import integrate
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
import ttkbootstrap as ttk
from ttkbootstrap.dialogs.dialogs import Querybox

from veremisfitting.line_fitting_model import fitting_model
from veremisfitting.select_gui import *
from veremisfitting.fitting_window_gui import *
from veremisfitting.data_loading_utils import *
from veremisfitting.modeling_utils import *
from veremisfitting.analysis_utils import *

from IPython import embed

class line_fitting_exec():
    def __init__(self, redshift = None, vac_or_air = 'air', seed = 42, n_jobs = -1, folder_name = None, file_name = None, line_select_method = 'gui', input_txt = None, 
                 fit_cont_order = 1, fit_window_gui = False, params_windows_gui = True, sigma_min = 30, sigma_max_e = 1200, sigma_max_a = 1500, 
                 fit_algorithm = 'leastsq'):
        """
        Constructor for the line_fitting_exec class that initializes the following class variables:
        
        c : float
            The speed of light in km/s.
        rng : numpy.random.Generator
            The random number generator.
        n_jobs: float
            The number of jobs to run in parallel. Default is -1 (use all processors).
        redshift : float
            The redshift of the galaxy.
        fit_cont_order : float
            The order of fitting local continuum level. Default is 1.
        fit_window_gui : bool
            Whether to interactively determine the fitting window, local continuum regions, and masking lines. Default is False.
        params_windows_gui : bool
            Whether to pop up the GUI window for interactively determine the initial parameter values and their corresponding ranges (for each iteration). Default is True (i.e., pop up the window).
        sigma_min : float
            The minimum velocity width (i.e., instrumental seeing) for each velocity component. Default is 30 km / s.
        sigma_max_e : float
            The maximum velocity width for each velocity component (for emissions). Default is 1200 km / s.
        sigma_max_a : float
            The maximum velocity width for each velocity component (for absorptions). Default is 1500 km / s.
        fit_algorithm : str
            The algorithm used for fitting (the well-tested ones include ’leastsq’: Levenberg-Marquardt (default), ’nelder’: Nelder-Mead, and ’powell’: Powell)
        """
        self.c = const.c.to('km/s').value
        # save the seed value and apply to the fitting_model function such that it has the same randomness
        self.seed = seed
        self.rng = np.random.default_rng(self.seed) 
        self.n_jobs = n_jobs
        if redshift == None:
            self.redshift = 0. # assume it is already rest-frame
        if redshift != None:
            self.redshift = redshift
        self.fit_cont_order = fit_cont_order
        self.fit_window_gui = fit_window_gui
        self.params_windows_gui = params_windows_gui
        self.sigma_limits = [sigma_min, sigma_max_e, sigma_max_a]
        self.fit_algorithm = fit_algorithm

        # Get the directory of the current script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the parent directory as the upper directory of the script directory
        self.parent_dir = os.path.abspath(os.path.join(self.script_dir, '..'))

        # get the rest-frame and observed-frame wavelength of each intended line
        # check whether the observed frame is in vacuum 'vac' or in 'air' (ADD any intended lines for fittings)
        if vac_or_air == 'air':
            self.elements_dict = read_wavelengths_from_file(os.path.join(self.parent_dir, 'doc', 'air_wavelengths.txt'), self.redshift)
        if vac_or_air == 'vac':
            self.elements_dict = read_wavelengths_from_file(os.path.join(self.parent_dir, 'doc', 'vac_wavelengths.txt'), self.redshift)
        self.wave_dict = merge_element_waves(self.elements_dict)

        # get the default half velocity width for fitting window
        self.half_v_width_window_dict = read_window_width_from_file(os.path.join(self.parent_dir, 'doc', 'default_window_vspace.txt'))

        # emission lines available for fitting
        self.emission_lines_lst = list(self.wave_dict.keys())

        # initialize a gui for users to select the lines to be fitted
        # it also allows users to choose whether to fix the velocity centroid or width during fittings
        if line_select_method == 'gui':
            self.selector = LineSelector(self.elements_dict)
            self.selected_lines, self.fitting_method, self.broad_wings_lines, self.double_gauss_lines, self.triple_gauss_lines, self.absorption_lines, self.fit_func_choices, self.fit_func_abs_choices = \
            self.selector.run() # selected lines and the selected fitting method

            # whether the second or the third emission components are broad or not
            self.double_gauss_broad = True if self.selector.double_gauss_broad == 'yes' else False 
            self.triple_gauss_broad = True if self.selector.triple_gauss_broad == 'yes' else False 

            # get the initial guess, range size for each parameter, and also the fixed ratios for selected amplitude pairs
            self.initial_guess_dict, self.param_range_dict, self.amps_ratio_dict = self.selector.params_dict, self.selector.params_range_dict, self.selector.amps_ratio_dict

        # users need to input a line selection text for this approach
        if line_select_method == 'txt':
            if self.params_windows_gui:
                line_select_pars = extract_line_pars(input_txt, return_amps_fixed_dict = False)
            else:
                line_select_pars, self.amps_ratio_dict = extract_line_pars(input_txt, return_amps_fixed_dict = True)
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
            # emission lines that need a Lorentzian function for fitting broad wings ("core" is fitted by Gaussians by default)
            self.fit_func_choices = {line: "Lorentzian" for line in line_select_pars['lorentz_bw_em_lines']}
            # absorption troughs that need a Lorentzian function for fitting broad wings (especially for Balmer lines)
            self.fit_func_abs_choices = {line: "Lorentzian" for line in line_select_pars['lorentz_bw_abs_lines']}

            # Call FitParamsWindow
            fit_window = FitParamsWindow(self.selected_lines, self.broad_wings_lines, self.triple_gauss_lines, self.absorption_lines, params_windows_gui = self.params_windows_gui)
            # get the initial guess, range size for each parameter, and also the fixed ratios for selected amplitude pairs
            if self.params_windows_gui:
                self.initial_guess_dict, self.param_range_dict, self.amps_ratio_dict = fit_window.run()
            else:
                self.initial_guess_dict, self.param_range_dict, _ = fit_window.run()

        # number of fixed velocity parameters for each self.fitting_method (times num of velocity components for each line profile)
        self.vel_fixed_num = {'Free fitting': 0, 'Fix velocity centroid': 1, 'Fix velocity width': 1, 'Fix velocity centroid and width': 2}

        # return the fixed line component names included in self.amps_ratio_dict 
        # since each key name in self.amps_ratio_dict is in the format of A_over_B, the fixed parameter for each key name is A here (for this example).
        self.amps_fixed_names = [part for ratio, value in self.amps_ratio_dict.items() for part in extract_key_parts_from_ratio(ratio)][::2] # [::2]/[1::2] to get the even/odd index; 

        # get the folder name from user input
        if folder_name != None:
            self.folder_name = folder_name 
        else:
            self.folder_name = Querybox.get_string(title="Input", prompt="Please enter a subfolder name for your saved results:", initialvalue="")
            if not self.folder_name: # when users forget to enter the folder name
                self.folder_name = "test_folder" # default folder name

        # get the name of the input fits file (excluding .fits)
        self.file_name = f"{self.folder_name}_{file_name}" if file_name != None else self.folder_name

        # initialize the line-continuum dict for all intended lines
        self.cont_line_dict = dict()

        # initialize the line mask dict for all masked lines
        self.lmsks_dict = dict()

    def extract_fit_window(self, wave, spec, espec, selected_line, indx):
        '''
        Extract and return velocity array, flux array (in velocity space), and error array (in velocity space)
        for the given selected line or multiplet.
        '''
        line_waves = self.wave_dict[selected_line]

        # Calculate the central wavelength for velocity transformation
        # For a multiplet, use the average of the wavelengths
        central_wave = np.mean(line_waves)

        nan_index = np.isnan(spec) | np.isnan(espec)  # nan-value index
        # copy of wave, spec, and espec
        wave_c = np.copy(wave[np.logical_not(nan_index)])
        spec_c = np.copy(spec[np.logical_not(nan_index)])
        espec_c = np.copy(espec[np.logical_not(nan_index)])

        # Determine the window size for fitting using the central wavelength
        half_v_width_window = self.half_v_width_window_dict[selected_line] # read the half velocity width of fitting window
        indx_lolim, indx_uplim = window_size_in_lam(half_v_width_window, central_wave, scale_factor=2., return_indx=True, wave_arr=wave_c)
        wave_fit = np.float64(wave_c[indx_lolim:indx_uplim+1])
        flux_fit = np.float64(spec_c[indx_lolim:indx_uplim+1])
        err_fit = np.float64(espec_c[indx_lolim:indx_uplim+1])

        # determine the local continuum and subtract it from the flux array
        cont_fit = local_cont_reg(wave_c, indx_lolim, indx_uplim, fraction = 0.2)
        
        # interactively determine the fitting window, local continuum regions, and masking lines
        if self.fit_window_gui:
            # initialize a gui for determining fitting window 
            if indx == 0:
                Fitting_Window = FittingWindow(wave_c, spec_c, folder_name = self.folder_name, line_name = selected_line, indx = indx)
                self.bokeh_session = Fitting_Window.session 
            else:
                Fitting_Window = FittingWindow(wave_c, spec_c, folder_name = self.folder_name, line_name = selected_line, indx = indx, bokeh_session = self.bokeh_session)
            # new boundary and line mask lists
            boundary, lmasks = Fitting_Window.run_process(central_wave, wave_c[indx_lolim], wave_c[indx_uplim], cont_fit[0][1], cont_fit[1][0], mask_lines = True)
            # extract new fit window
            new_fit_window_indx = np.where((wave_c >= boundary[0]) & (wave_c <= boundary[1]))
            cont_fit = [[boundary[0], boundary[2]],[boundary[3], boundary[1]]]
        else: # find if there are any local lmask file
            Fitting_Window = FittingWindow(wave_c, spec_c, folder_name = self.folder_name, line_name = selected_line, indx = 1) # a random indx number (!= 0) to not start bokeh server
            lmasks, _ = Fitting_Window.find_local_lmsk_file(selected_line)
            cont_dict, _ = Fitting_Window.find_local_cont_file(selected_line)
            if len(cont_dict) == 1:
                for ct in cont_dict:
                    x1, x2, x3, x4 = ct['x1'], ct['x2'], ct['x3'], ct['x4']
                cont_fit = [[x1, x3],[x4, x2]]
            else:
                x1, x2 = cont_fit[0][0], cont_fit[1][1]
            # extract new fit window
            new_fit_window_indx = np.where((wave_c >= x1) & (wave_c <= x2))
        # new wave, flux, and err arrays
        wave_fit = np.float64(wave_c[new_fit_window_indx])
        flux_fit = np.float64(spec_c[new_fit_window_indx])
        err_fit = np.float64(espec_c[new_fit_window_indx])

        # mask selected lines
        if len(lmasks) >= 1:
            self.lmsks_dict[selected_line] = lmasks # save for later plotting (show the masked regions)
            # save the unmasked raw data for later plotting 
            wave_fit_c = np.copy(wave_fit)
            flux_fit_c = np.copy(flux_fit)
            err_fit_c = np.copy(err_fit)
            # masked the raw data for LMFIT fitting
            for ml in lmasks:
                wave_fit[np.where((wave_fit >= ml['w0'])&(wave_fit <= ml['w1']))] = 0
                wave_fit = np.ma.masked_values(wave_fit, 0)
                mask = wave_fit.mask
                flux_fit = np.ma.masked_array(flux_fit, mask=mask).compressed()
                err_fit = np.ma.masked_array(err_fit, mask=mask).compressed()
                wave_fit = wave_fit.compressed()
        
        # fit the local continuum level and extract it from the flux array
        cont_f_fit, cont_f_fit_err, linecoeff, covar = region_around_line(wave_fit, flux_fit, cont_fit, order = self.fit_cont_order)[-4:]
        self.cont_line_dict[selected_line] = np.array([cont_f_fit, cont_f_fit_err])
        # local-continuum-subtracted spectrum
        flux_sc_fit = flux_fit - cont_f_fit
        # error propagation
        err_sc_fit = np.sqrt((err_fit)**2 + (cont_f_fit_err)**2)

        # create a dictionary for best-fit params for continuum fit
        if not hasattr(self, 'cont_best_fit_params'):
            self.cont_best_fit_params = dict()
        # save the best-fit polynomial coefficient and its error into this dictionary
        cont_params = []
        for iorder in range(self.fit_cont_order + 1):
            # coeff_name = f"p_{iorder}"
            indx = self.fit_cont_order - iorder
            cont_params.append([linecoeff[indx], np.sqrt(covar[indx, indx])])
        self.cont_best_fit_params[selected_line] = np.array(cont_params)

        # Choose the appropriate line wave based on the length of line_waves
        chosen_line_wave = line_waves[0] if len(line_waves) <= 2 else line_waves[1]

        # Transform wavelength array to velocity array for each line
        v_fit = [(wave_fit / w - 1) * self.c for w in line_waves]
        # Transform continuum-subtracted flux array to velocity space
        flux_sc_v_fit = [flux_sc_fit * chosen_line_wave / self.c]
        # Transform continuum-subtracted error array to velocity space
        err_sc_v_fit = [err_sc_fit * chosen_line_wave / self.c]
        # concatenate velocity, flux, and error arrays
        # result_fit = v_fit + flux_sc_v_fit + err_sc_v_fit
        
        # check whether there are masked regions
        if len(lmasks) >= 1: # if yes, then append the unmasked velocity and flux (no continuum subtracted) arrays 
            v_fit_c = [(wave_fit_c / w - 1) * self.c for w in line_waves]
            flux_v_fit_c = [flux_fit_c * chosen_line_wave / self.c]
            err_v_fit_c = [err_fit_c * chosen_line_wave / self.c]
            result_fit = v_fit + flux_sc_v_fit + err_sc_v_fit + v_fit_c + flux_v_fit_c + err_v_fit_c
        else:
            # if no, then append the masked velocity and flux (no continuum subtracted) arrays again
            flux_v_fit = [flux_fit * chosen_line_wave / self.c]
            err_v_fit = [err_fit * chosen_line_wave / self.c]
            result_fit = v_fit + flux_sc_v_fit + err_sc_v_fit + v_fit + flux_v_fit + err_v_fit

        return result_fit

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
        # step 1: obtain the velocity array, flux array (in velocity space), and error array (in velocity space)
        # initialize the velocity, flux, and flux_err dictionaries
        self.velocity_dict = dict()
        self.flux_v_dict = dict()
        self.err_v_dict = dict()
        self.velocity_c_dict = dict() # unmasked version
        self.flux_v_c_dict = dict() # unmasked version (no continuum subtracted)
        self.err_v_c_dict = dict() # unmasked version (no continuum subtracted)

        # Iterate through each selected line and return the corresponding extracted velocity, flux, and error arrays
        for indx, line in enumerate(self.selected_lines):
            if '&' in line:  # Special case for multilet
                multilet_lines = split_multilet_line(line)
                # doublet
                if len(multilet_lines) == 2:
                    v_arr, v_arr_2, flux_v_arr, err_v_arr, v_c_arr, v_c_arr_2, flux_v_c_arr, err_v_c_arr = self.extract_fit_window(wave, spec, espec, line, indx)
                    self.velocity_dict[line] = np.array([v_arr, v_arr_2])
                    self.velocity_c_dict[line] = np.array([v_c_arr, v_c_arr_2])
                # triplet
                if len(multilet_lines) == 3:
                    v_arr, v_arr_2, v_arr_3, flux_v_arr, err_v_arr, v_c_arr, v_c_arr_2, v_c_arr_3, flux_v_c_arr, err_v_c_arr = self.extract_fit_window(wave, spec, espec, line, indx)
                    self.velocity_dict[line] = np.array([v_arr, v_arr_2, v_arr_3])
                    self.velocity_c_dict[line] = np.array([v_c_arr, v_c_arr_2, v_c_arr_3])
            else:  # General case
                v_arr, flux_v_arr, err_v_arr, v_c_arr, flux_v_c_arr, err_v_c_arr = self.extract_fit_window(wave, spec, espec, line, indx)
                self.velocity_dict[line] = v_arr
                self.velocity_c_dict[line] = v_c_arr
            self.flux_v_dict[line] = flux_v_arr
            self.err_v_dict[line] = err_v_arr
            self.flux_v_c_dict[line] = flux_v_c_arr
            self.err_v_c_dict[line] = err_v_c_arr
                    
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
        for line in self.selected_lines:
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
                     self.broad_wings_lines, self.double_gauss_lines, self.triple_gauss_lines, self.double_gauss_broad, self.triple_gauss_broad, self.fitting_method, 
                     self.fit_func_choices, self.fit_func_abs_choices, self.sigma_limits, self.fit_algorithm)
        self.line_fit_model = fitting_model(seed = self.seed, n_jobs = self.n_jobs)
        best_model, residual_dict, best_chi2 = self.line_fit_model.fitting_all_lines(input_arr, n_iteration = n_iteration)
        return best_model, residual_dict, best_chi2


    def all_lines_result(self, wave, spec, espec, n_iteration = 1000, get_flux = False, save_flux_table = False, get_ew = True, 
                         save_ew_table = False, get_error = False, save_par_table = False, save_stats_table = False, save_cont_params_table = False):
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
        save_stats_table : bool, optional
            If True, save the best-fitting statistics, including chi2, redchi2, AIC, BIC, etc. for each selected line. Default is False.
        save_cont_params_table : bool, optional
            If True, save the best-fitting parameters for continuum fit for each line. Default is False.
        Returns:
        --------
        """
        # return the best-fitting models
        self.model_dict, self.residual_dict, self.best_chi2 = self.fitting_input_params(wave, spec, espec, n_iteration = n_iteration)

        # best-fitting velocity centers and widths
        self.x0_e = self.line_fit_model.best_param_dict["center_e"]
        self.sigma_e = self.line_fit_model.best_param_dict["sigma_e"]
        self.x0_a = self.line_fit_model.best_param_dict.get("center_a", None)
        self.sigma_a = self.line_fit_model.best_param_dict.get("sigma_a", None)
        self.x0_b = self.line_fit_model.best_param_dict.get("center_b", None)
        self.sigma_b = self.line_fit_model.best_param_dict.get("sigma_b", None)
        self.x0_b2 = self.line_fit_model.best_param_dict.get("center_b2", None)
        self.sigma_b2 = self.line_fit_model.best_param_dict.get("sigma_b2", None)

        # create a dict for x0 and sigma respectively
        self.x0_dict = {'x0_e': self.x0_e, 'x0_b': self.x0_b, 'x0_b2': self.x0_b2, 'x0_a': self.x0_a}
        self.sigma_dict = {'sigma_e': self.sigma_e, 'sigma_b': self.sigma_b, 'sigma_b2': self.sigma_b2, 'sigma_a': self.sigma_a}

        # get the dict that contains the best-fitting amplitudes for all lines
        self.amps_dict = self.line_fit_model.best_amps

        # initialize the dicts for residuals, chi-square, and best-fitting models
        self.redchi2_dict = dict()
        # whether to initialize the dicts for statistics 
        if save_stats_table:
            self.ndata_dict = dict() # number of data points 
            self.nvarys_dict = dict() # number of variables in fit
            self.dof_dict = dict() # degree of freedom
            self.chi2_dict = dict() # chi-square: np.sum(residual**2)
            self.aic_lmfit_dict = dict() # lmfit aic
            self.bic_lmfit_dict = dict() # lmfit bic
            self.aic_duvet_dict = dict() # duvet aic
            self.bic_duvet_dict = dict() # duvet bic
        # model for narrow and broad emission components
        self.best_model_n_dict = self.line_fit_model.best_model_n_dict
        if self.broad_wings_lines:
            self.best_model_b_dict = self.line_fit_model.best_model_b_dict
            if self.triple_gauss_lines:
                self.best_model_b2_dict = self.line_fit_model.best_model_b2_dict
        # model for emission and absorption components
        if self.absorption_lines:
            self.best_model_ab_dict = self.line_fit_model.best_model_ab_dict

        # iterate through each line and return their best-fitting stats
        for line in self.selected_lines:
            # number of parameters for this line
            params_num = self.line_fit_model.params_num_dict[line]
            # append the line chi2 to the chi2 dict
            ndata = len(self.residual_dict[line]) # number of data points
            nfixs_amp = num_fix_amps(line, self.amps_fixed_names) # number of fixed amplitude variables in fit
            nfixs_vel = num_fix_vels(line, self.fitting_method, self.vel_fixed_num, self.broad_wings_lines, self.triple_gauss_lines, self.absorption_lines) # number of fixed velocity variables in fit
            nvarys = params_num - nfixs_amp - nfixs_vel # number of variables in fit
            dof = ndata - nvarys # degree of freedom
            chisqr = np.sum(self.residual_dict[line]**2)
            self.redchi2_dict[line] = chisqr / dof
            if save_stats_table:
                self.ndata_dict[line] = ndata
                self.nvarys_dict[line] = nvarys
                self.dof_dict[line] = dof 
                self.chi2_dict[line] = chisqr
                self.aic_lmfit_dict[line] = aic_lmfit(chisqr, nvarys, ndata)
                self.bic_lmfit_dict[line] = bic_lmfit(chisqr, nvarys, ndata)
                self.aic_duvet_dict[line] = aic_duvet(chisqr, nvarys)
                self.bic_duvet_dict[line] = bic_duvet(chisqr, nvarys, ndata)

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

        for line in self.selected_lines:
            if '&' in line:  # Special case for multilet that should be fitted together
                multilet_lines = split_multilet_line(line)
                for i, l in enumerate(multilet_lines):
                    self.sigma_w_dict[l] = self.sigma_e * self.wave_dict[line][i] / self.c
                    self.lambda_w_dict[l] = self.wave_dict[line][i] * (1. + (self.x0_e / self.c))
                    self.flux_dict[l] = np.abs(self.amps_dict[l.split(' ')[1]]*self.sigma_e*np.sqrt(2*np.pi))
                    # multi emission component
                    if l in self.broad_wings_lines:
                        broad_amp = self.amps_dict[f"{l.split(' ')[1]}_b"]
                        self.sigma_b_w_dict[l] = self.sigma_b * self.wave_dict[line][i] / self.c
                        self.lambda_b_w_dict[l] = self.wave_dict[line][i] * (1. + (self.x0_b / self.c))
                        if (l in self.fit_func_choices.keys()) and (self.fit_func_choices[l] == 'Lorentzian') and (l not in self.triple_gauss_lines):
                            self.flux_b_dict[l] = lorentz_integral_analy(broad_amp, self.sigma_b).real
                        else:
                            self.flux_b_dict[l] = np.abs(broad_amp*self.sigma_b*np.sqrt(2*np.pi)) 
                        if l in self.triple_gauss_lines:
                            broad_amp2 = self.amps_dict[f"{l.split(' ')[1]}_b2"]
                            self.sigma_b2_w_dict[l] = self.sigma_b2 * self.wave_dict[line][i] / self.c
                            self.lambda_b2_w_dict[l] = self.wave_dict[line][i] * (1. + (self.x0_b2 / self.c))
                            if (l in self.fit_func_choices.keys()) and (self.fit_func_choices[l] == 'Lorentzian'):
                                self.flux_b2_dict[l] = lorentz_integral_analy(broad_amp2, self.sigma_b2).real
                            else:
                                self.flux_b2_dict[l] = np.abs(broad_amp2*self.sigma_b2*np.sqrt(2*np.pi)) 
                    # absorption component
                    if l in self.absorption_lines:
                        abs_amp = self.amps_dict[f"{l.split(' ')[1]}_abs"]
                        self.sigma_abs_w_dict[l] = self.sigma_a * self.wave_dict[line][i] / self.c
                        self.lambda_abs_w_dict[l] = self.wave_dict[line][i] * (1. + (self.x0_a / self.c)) 
                        if (line in self.fit_func_abs_choices.keys()) and (self.fit_func_abs_choices[l] == 'Lorentzian'):
                            self.flux_abs_dict[l] = np.abs(lorentz_integral_analy(abs_amp, self.sigma_a).real)
                        else:
                            self.flux_abs_dict[l] = np.abs(abs_amp*self.sigma_a*np.sqrt(2*np.pi))

            else: # General case
                amp = self.amps_dict[line.split(' ')[1]]
                self.sigma_w_dict[line] = self.sigma_e * self.wave_dict[line][0] / self.c
                self.lambda_w_dict[line] = self.wave_dict[line][0] * (1. + (self.x0_e / self.c))
                self.flux_dict[line] = np.abs(amp*self.sigma_e*np.sqrt(2*np.pi))

                if (line in self.broad_wings_lines):
                    broad_amp = self.amps_dict[f"{line.split(' ')[1]}_b"]
                    self.sigma_b_w_dict[line] = self.sigma_b * self.wave_dict[line][0] / self.c
                    self.lambda_b_w_dict[line] = self.wave_dict[line][0] * (1. + (self.x0_b / self.c))
                    if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian'):
                        self.flux_b_dict[line] = lorentz_integral_analy(broad_amp, self.sigma_b).real
                    else:
                        self.flux_b_dict[line] = np.abs(broad_amp*self.sigma_b*np.sqrt(2*np.pi)) 
                    if line in self.triple_gauss_lines:
                        broad_amp2 = self.amps_dict[f"{line.split(' ')[1]}_b2"]
                        self.sigma_b2_w_dict[line] = self.sigma_b2 * self.wave_dict[line][0] / self.c
                        self.lambda_b2_w_dict[line] = self.wave_dict[line][0] * (1. + (self.x0_b2 / self.c))
                        if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian'):
                            self.flux_b2_dict[line] = lorentz_integral_analy(broad_amp2, self.sigma_b2).real
                        else:
                            self.flux_b2_dict[line] = np.abs(broad_amp2*self.sigma_b2*np.sqrt(2*np.pi)) 
                if (line in self.absorption_lines):
                    abs_amp = self.amps_dict[f"{line.split(' ')[1]}_abs"]
                    self.sigma_abs_w_dict[line] = self.sigma_a * self.wave_dict[line][0] / self.c
                    self.lambda_abs_w_dict[line] = self.wave_dict[line][0] * (1. + (self.x0_a / self.c)) 
                    if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian'):
                        self.flux_abs_dict[line] = np.abs(lorentz_integral_analy(abs_amp, self.sigma_a).real)
                    else:
                        self.flux_abs_dict[line] = np.abs(abs_amp*self.sigma_a*np.sqrt(2*np.pi))

        # return the error of each line flux derived above
        if get_error:
            # error of the velocity centers and widths
            self.x0_e_err = self.line_fit_model.best_res.params["center_e"].stderr
            self.sigma_e_err = self.line_fit_model.best_res.params["sigma_e"].stderr
            self.x0_dict['x0_e_err'] = self.x0_e_err
            self.sigma_dict['sigma_e_err'] = self.sigma_e_err
            if self.broad_wings_lines:
                self.x0_b_err = self.line_fit_model.best_res.params["center_b"].stderr
                self.sigma_b_err = self.line_fit_model.best_res.params["sigma_b"].stderr
                self.x0_dict['x0_b_err'] = self.x0_b_err
                self.sigma_dict['sigma_b_err'] = self.sigma_b_err
                if self.triple_gauss_lines:
                    self.x0_b2_err = self.line_fit_model.best_res.params["center_b2"].stderr
                    self.sigma_b2_err = self.line_fit_model.best_res.params["sigma_b2"].stderr
                    self.x0_dict['x0_b2_err'] = self.x0_b2_err
                    self.sigma_dict['sigma_b2_err'] = self.sigma_b2_err
            if self.absorption_lines:
                self.x0_a_err = self.line_fit_model.best_res.params["center_a"].stderr
                self.sigma_a_err = self.line_fit_model.best_res.params["sigma_a"].stderr
                self.x0_dict['x0_a_err'] = self.x0_a_err
                self.sigma_dict['sigma_a_err'] = self.sigma_a_err
            # error of each amplitude
            for line in self.selected_lines:
                if '&' in line:  # Special case for doublet that should be fitted together
                    multilet_lines = split_multilet_line(line)
                    for i, l in enumerate(multilet_lines):
                        amp = self.amps_dict[l.split(' ')[1]]
                        amp_err = self.line_fit_model.best_res.params[f"amp_{l.split(' ')[1]}"].stderr
                        self.amps_dict[f"{l.split(' ')[1]}_err"] = amp_err
                        try:
                            self.flux_err_dict[l] = np.sqrt((np.sqrt(2*np.pi)*amp*self.sigma_e_err)**2 + (np.sqrt(2*np.pi)*self.sigma_e*amp_err)**2)
                        except TypeError:
                            self.flux_err_dict[l] = np.nan
                        # multi emission component
                        if l in self.broad_wings_lines:
                            broad_amp = self.amps_dict[f"{l.split(' ')[1]}_b"]
                            broad_amp_err = self.line_fit_model.best_res.params[f"amp_{l.split(' ')[1]}_b"].stderr
                            self.amps_dict[f"{l.split(' ')[1]}_b_err"] = broad_amp_err
                            try:
                                if (l in self.fit_func_choices.keys()) and (self.fit_func_choices[l] == 'Lorentzian') and (l not in self.triple_gauss_lines):
                                    # self.flux_b_err_dict[l] = run_mc.line_flux_lorentz_MC(100, self.x0_b, self.sigma_b, broad_amp, self.x0_b_err, self.sigma_b_err, broad_amp_err)[0] # MC simulation 
                                    self.flux_b_err_dict[l] = lorentz_integral_analy_err(broad_amp, self.sigma_b, broad_amp_err, self.sigma_b_err).real # analytical 
                                else:
                                    self.flux_b_err_dict[l] = np.sqrt((np.sqrt(2*np.pi)*broad_amp*self.sigma_b_err)**2 + (np.sqrt(2*np.pi)*self.sigma_b*broad_amp_err)**2)
                            except TypeError:
                                self.flux_b_err_dict[l] = np.nan
                            if l in self.triple_gauss_lines:
                                broad_amp2 = self.amps_dict[f"{l.split(' ')[1]}_b2"]
                                broad_amp2_err = self.line_fit_model.best_res.params[f"amp_{l.split(' ')[1]}_b2"].stderr
                                self.amps_dict[f"{l.split(' ')[1]}_b2_err"] = broad_amp2_err
                                try:
                                    if (l in self.fit_func_choices.keys()) and (self.fit_func_choices[l] == 'Lorentzian'):
                                        # self.flux_b2_err_dict[l] = run_mc.line_flux_lorentz_MC(100, self.x0_b2, self.sigma_b2, broad_amp2, self.x0_b2_err, self.sigma_b2_err, broad_amp2_err)[0] # MC simulation
                                        self.flux_b2_err_dict[l] = lorentz_integral_analy_err(broad_amp2, self.sigma_b2, broad_amp2_err, self.sigma_b2_err).real # analytical
                                    else:
                                        self.flux_b2_err_dict[l] = np.sqrt((np.sqrt(2*np.pi)*broad_amp2*self.sigma_b2_err)**2 + (np.sqrt(2*np.pi)*self.sigma_b2*broad_amp2_err)**2)
                                except TypeError:
                                    self.flux_b2_err_dict[l] = np.nan
                        # absorption component
                        if l in self.absorption_lines:
                            abs_amp = self.amps_dict[f"{l.split(' ')[1]}_abs"]
                            abs_amp_err = self.line_fit_model.best_res.params[f"amp_{l.split(' ')[1]}_abs"].stderr
                            self.amps_dict[f"{l.split(' ')[1]}_abs_err"] = abs_amp_err
                            try:
                                if (l in self.fit_func_abs_choices.keys()) and (self.fit_func_abs_choices[l] == 'Lorentzian'):
                                    self.flux_abs_err_dict[l] = lorentz_integral_analy_err(abs_amp, self.sigma_a, abs_amp_err, self.sigma_a_err).real # analytical
                                else:
                                    self.flux_abs_err_dict[l] = np.sqrt((np.sqrt(2*np.pi)*abs_amp*self.sigma_a_err)**2 + (np.sqrt(2*np.pi)*self.sigma_a*abs_amp_err)**2)
                            except TypeError:
                                self.flux_abs_err_dict[l] = np.nan
                else: # General case
                    # flux error of the narrow emission component
                    amp = self.amps_dict[line.split(' ')[1]]
                    amp_err = self.line_fit_model.best_res.params[f"amp_{line.split(' ')[1]}"].stderr
                    self.amps_dict[f"{line.split(' ')[1]}_err"] = amp_err
                    try:
                        self.flux_err_dict[line] = np.sqrt((np.sqrt(2*np.pi)*amp*self.sigma_e_err)**2 + (np.sqrt(2*np.pi)*self.sigma_e*amp_err)**2)
                    except TypeError:
                        self.flux_err_dict[line] = np.nan
                    # flux error of the broad emission component
                    if (line in self.broad_wings_lines):
                        broad_amp = self.amps_dict[f"{line.split(' ')[1]}_b"]
                        broad_amp_err = self.line_fit_model.best_res.params[f"amp_{line.split(' ')[1]}_b"].stderr
                        self.amps_dict[f"{line.split(' ')[1]}_b_err"] = broad_amp_err
                        try:
                            if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian'):
                                # self.flux_b_err_dict[line] = run_mc.line_flux_lorentz_MC(100, self.x0_b, self.sigma_b, broad_amp, self.x0_b_err, self.sigma_b_err, broad_amp_err)[0] # MC simulation 
                                self.flux_b_err_dict[line] = lorentz_integral_analy_err(broad_amp, self.sigma_b, broad_amp_err, self.sigma_b_err).real # analytical 
                            else:
                                self.flux_b_err_dict[line] = np.sqrt((np.sqrt(2*np.pi)*broad_amp*self.sigma_b_err)**2 + (np.sqrt(2*np.pi)*self.sigma_b*broad_amp_err)**2)
                        except TypeError:
                            self.flux_b_err_dict[line] = np.nan
                        if line in self.triple_gauss_lines:
                            broad_amp2 = self.amps_dict[f"{line.split(' ')[1]}_b2"]
                            broad_amp2_err = self.line_fit_model.best_res.params[f"amp_{line.split(' ')[1]}_b2"].stderr
                            self.amps_dict[f"{line.split(' ')[1]}_b2_err"] = broad_amp2_err
                            try:
                                if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian'):
                                    # self.flux_b2_err_dict[line] = run_mc.line_flux_lorentz_MC(100, self.x0_b2, self.sigma_b2, broad_amp2, self.x0_b2_err, self.sigma_b2_err, broad_amp2_err)[0]
                                    self.flux_b2_err_dict[line] = lorentz_integral_analy_err(broad_amp2, self.sigma_b2, broad_amp2_err, self.sigma_b2_err).real # analytical
                                else:
                                    self.flux_b2_err_dict[line] = np.sqrt((np.sqrt(2*np.pi)*broad_amp2*self.sigma_b2_err)**2 + (np.sqrt(2*np.pi)*self.sigma_b2*broad_amp2_err)**2)
                            except TypeError:
                                self.flux_b2_err_dict[line] = np.nan
                    # flux error of the absorption component
                    if (line in self.absorption_lines):
                        abs_amp = self.amps_dict[f"{line.split(' ')[1]}_abs"]
                        abs_amp_err = self.line_fit_model.best_res.params[f"amp_{line.split(' ')[1]}_abs"].stderr
                        self.amps_dict[f"{line.split(' ')[1]}_abs_err"] = abs_amp_err
                        try:
                            if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian'):
                                self.flux_abs_err_dict[line] = lorentz_integral_analy_err(abs_amp, self.sigma_a, abs_amp_err, self.sigma_a_err).real # analytical
                            else:
                                self.flux_abs_err_dict[line] = np.sqrt((np.sqrt(2*np.pi)*abs_amp*self.sigma_a_err)**2 + (np.sqrt(2*np.pi)*self.sigma_a*abs_amp_err)**2)
                        except TypeError:
                            self.flux_abs_err_dict[line] = np.nan

        # return the ew of each balmer line in wavelength space
        if get_ew:
            self.find_ew(save_ew_table = save_ew_table)

        # whether to save the parameter table 
        if save_par_table:
            self.save_par_pd_table()

        # whether to save the stats table
        if save_stats_table:
            self.save_stats_pd_table()

        # whether to save the best-fit parameters for continuum fit table
        if save_cont_params_table:
            self.save_cont_param_pd_table()
        
        # whether to return the flux (and err) dict
        if get_flux:
            # whether to save the flux table
            if save_flux_table:
                self.save_flux_pd_table()
            if get_error:
                return (self.flux_dict, self.flux_err_dict)
            if not get_error:
                return self.flux_dict

        return self.line_fit_model.best_params

    def save_cont_param_pd_table(self):
        '''
        Saves a Pandas DataFrame containing the best-fitting parameters of continuum fit to a CSV file.
        The DataFrame will have columns for each coefficient and its standard deviation,
        for a polynomial fit of a given order.
        '''
        # Initialize an empty list to collect each row (each line's fit parameters) for the DataFrame
        rows_list = []
        index_list = []  # To keep track of the row names (selected line names)
        
        for line_name, params in self.cont_best_fit_params.items():
            # 'params' should be a list of [coefficient, std] pairs for each order
            row = {}
            for iorder, (coeff, std) in enumerate(params):
                coeff_name = f"p{iorder}"
                std_name = f"{coeff_name}_std"
                row[coeff_name] = coeff
                row[std_name] = std
            rows_list.append(row)
            index_list.append(line_name)
        
        # Create the DataFrame from the list of rows
        self.cont_df = pd.DataFrame(rows_list, index=index_list)
        
        # Print the DataFrame to check
        print(self.cont_df)

        # Define the parent directory for the continuum parameters table
        directory = f"cont_tables/{self.folder_name}/"
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the whole DataFrame to a CSV file in the directory
        self.cont_df.to_csv(os.path.join(directory, self.file_name + '_cont.csv'))

    def save_stats_pd_table(self):
        '''
        Saves a Pandas DataFrame containing the statistics of each line to a CSV file.
        '''
        self.stats_df = pd.DataFrame({
            'ndata': self.ndata_dict if hasattr(self, 'ndata_dict') else {},
            'nvarys': self.nvarys_dict if hasattr(self, 'nvarys_dict') else {},
            'dof': self.dof_dict if hasattr(self, 'dof_dict') else {},
            'chi2': self.chi2_dict if hasattr(self, 'chi2_dict') else {},
            'redchi2': self.redchi2_dict if hasattr(self, 'redchi2_dict') else {},
            'aic_lmfit': self.aic_lmfit_dict if hasattr(self, 'aic_lmfit_dict') else {},
            'aic_duvet': self.aic_duvet_dict if hasattr(self, 'aic_duvet_dict') else {},
            'bic_lmfit': self.bic_lmfit_dict if hasattr(self, 'bic_lmfit_dict') else {},
            'bic_duvet': self.bic_duvet_dict if hasattr(self, 'bic_duvet_dict') else {}
        })
        print(self.stats_df)

        # Define the parent directory for the flux table
        directory = f"stats_tables/{self.folder_name}/"
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the whole DataFrame to a CSV file in the directory
        self.stats_df.to_csv(os.path.join(directory, self.file_name + '_stats.csv'))


    def save_par_pd_table(self):
        '''
        Saves a Pandas DataFrame containing the best-fitting parameters of each line to a CSV file.
        '''
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
        directory = f"parameter_tables/{self.folder_name}/"
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the whole DataFrame to a CSV file in the directory
        self.par_df.to_csv(os.path.join(directory, self.file_name + '_parameters.csv'))

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
        directory = f"flux_tables/{self.folder_name}/"
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the whole DataFrame to a CSV file in the directory
        self.flux_df.to_csv(os.path.join(directory, self.file_name + '_flux.csv'))

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
        directory = f"ew_tables/{self.folder_name}/"
        # Create parent directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the whole DataFrame to a CSV file in the parent directory
        self.ew_df.to_csv(os.path.join(directory, self.file_name + '_ew.csv'))


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
        for line in self.selected_lines:
            # doublet that needs to be fitted together
            if '&' in line:  # Special case for doublet that should be fitted together
                multilet_lines = split_multilet_line(line)
                for ii, l in enumerate(multilet_lines):
                    # derive the ew for the combined line profile
                    self.model_w = self.model_dict[l] * self.c / self.wave_dict[line][ii]
                    wave_line = ((self.velocity_dict[line][ii] / self.c) + 1.) * self.wave_dict[line][ii] 
                    flux_all = (-self.model_w) / self.cont_line_dict[line][0]
                    ew_all, ew_all_err = calc_ew_with_err(self.model_w, wave_line, self.sigma_w_dict[l], self.cont_line_dict[line][0])
                    self.ew_all_dict[l] = ew_all
                    self.ew_all_err_dict[l] = ew_all_err

                    # derive the ew for the narrow and broad components of the line profile
                    if l in self.broad_wings_lines:
                        # first emission component
                        self.best_model_n_w = self.best_model_n_dict[l] * self.c / self.wave_dict[line][ii]
                        ew_e, ew_e_err = calc_ew_with_err(self.best_model_n_w, wave_line, self.sigma_w_dict[l], self.cont_line_dict[line][0])
                        self.ew_e_dict[l] = ew_e
                        self.ew_e_err_dict[l] = ew_e_err
                        # second emission component
                        self.best_model_b_w = self.best_model_b_dict[l] * self.c / self.wave_dict[line][ii]
                        guass_or_lorentz = 'lorentz' if (l in self.fit_func_choices.keys()) and (self.fit_func_choices[l] == 'Lorentzian') and (l not in self.triple_gauss_lines) else 'gauss'
                        ew_b, ew_b_err = calc_ew_with_err(self.best_model_b_w, wave_line, self.sigma_b_w_dict[l], self.cont_line_dict[line][0], guass_or_lorentz = guass_or_lorentz)
                        self.ew_b_dict[l] = ew_b
                        self.ew_b_err_dict[l] = ew_b_err
                        # if there is a third emission component
                        if line in self.triple_gauss_lines:
                            self.best_model_b2_w = self.best_model_b2_dict[l] * self.c / self.wave_dict[line][ii]
                            guass_or_lorentz = 'lorentz' if (l in self.fit_func_choices.keys()) and (self.fit_func_choices[l] == 'Lorentzian') else 'gauss'
                            ew_b2, ew_b2_err = calc_ew_with_err(self.best_model_b2_w, wave_line, self.sigma_b2_w_dict[l], self.cont_line_dict[line][0], guass_or_lorentz = guass_or_lorentz)
                            self.ew_b2_dict[l] = ew_b2
                            self.ew_b2_err_dict[l] = ew_b2_err
                    # derive the ew for the absorption component of the line profile
                    if l in self.absorption_lines:
                        self.best_model_ab_w = self.best_model_ab_dict[l] * self.c / self.wave_dict[line][ii]
                        guass_or_lorentz = 'lorentz' if (l in self.fit_func_abs_choices.keys()) and (self.fit_func_abs_choices[l] == 'Lorentzian') else 'gauss'
                        ew_a, ew_a_err = calc_ew_with_err(self.best_model_ab_w, wave_line, self.sigma_abs_w_dict[l], self.cont_line_dict[line][0], guass_or_lorentz = guass_or_lorentz)
                        self.ew_abs_dict[l] = ew_a
                        self.ew_abs_err_dict[l] = ew_a_err

            # single line profile
            else: # General case
                # derive the ew for the combined line profile
                self.model_w = self.model_dict[line] * self.c / self.wave_dict[line][-1]
                wave_line = ((self.velocity_dict[line] / self.c) + 1.) * self.wave_dict[line][-1] 
                flux_all = (-self.model_w) / self.cont_line_dict[line][0]
                ew_all, ew_all_err = calc_ew_with_err(self.model_w, wave_line, self.sigma_w_dict[line], self.cont_line_dict[line][0])
                self.ew_all_dict[line] = ew_all
                self.ew_all_err_dict[line] = ew_all_err
                # derive the ew for the absorption and emission components of the line profile
                if (line in self.absorption_lines):
                    self.best_model_ab_w = self.best_model_ab_dict[line] * self.c / self.wave_dict[line][-1]
                    guass_or_lorentz = 'lorentz' if (line in self.fit_func_abs_choices.keys()) and (self.fit_func_abs_choices[line] == 'Lorentzian') else 'gauss'
                    ew_a, ew_a_err = calc_ew_with_err(self.best_model_ab_w, wave_line, self.sigma_abs_w_dict[line], self.cont_line_dict[line][0], guass_or_lorentz = guass_or_lorentz)
                    self.ew_abs_dict[line] = ew_a
                    self.ew_abs_err_dict[line] = ew_a_err
                # derive the ew for the narrow and broad components of the line profile
                if (line in self.broad_wings_lines):
                    self.best_model_n_w = self.best_model_n_dict[line] * self.c / self.wave_dict[line][-1]
                    # first emission component
                    ew_e, ew_e_err = calc_ew_with_err(self.best_model_n_w, wave_line, self.sigma_w_dict[line], self.cont_line_dict[line][0])
                    self.ew_e_dict[line] = ew_e
                    self.ew_e_err_dict[line] = ew_e_err
                    # second emission component
                    self.best_model_b_w = self.best_model_b_dict[line] * self.c / self.wave_dict[line][-1]
                    guass_or_lorentz = 'lorentz' if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian') and (line not in self.triple_gauss_lines) else 'gauss'
                    ew_b, ew_b_err = calc_ew_with_err(self.best_model_b_w, wave_line, self.sigma_b_w_dict[line], self.cont_line_dict[line][0], guass_or_lorentz = guass_or_lorentz)
                    self.ew_b_dict[line] = ew_b
                    self.ew_b_err_dict[line] = ew_b_err
                    # if there is a third emission component
                    if line in self.triple_gauss_lines:
                        self.best_model_b2_w = self.best_model_b2_dict[line] * self.c / self.wave_dict[line][-1]
                        guass_or_lorentz = 'lorentz' if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian') else 'gauss'
                        ew_b2, ew_b2_err = calc_ew_with_err(self.best_model_b2_w, wave_line, self.sigma_b2_w_dict[line], self.cont_line_dict[line][0], guass_or_lorentz = guass_or_lorentz)
                        self.ew_b2_dict[line] = ew_b2
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
        rc('text', usetex = False) # usetex = True for font in latex format
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
        self.flux_plus_cont_v_dict_2 = dict()
        self.model_plus_cont_v_dict = dict()
        # whether include broad_wings_lines
        if self.broad_wings_lines:
            self.best_model_plus_cont_n_v_dict = dict()
            self.best_model_plus_cont_b_v_dict = dict()
            if self.triple_gauss_lines:
                self.best_model_plus_cont_b2_v_dict = dict()
        # whether include absorption_lines
        if self.absorption_lines:
            self.best_model_plus_cont_ab_v_dict = dict()

        for line in self.selected_lines:
            # Choose the appropriate line wave based on the length of line_waves
            line_waves = self.wave_dict[line]
            chosen_line_wave = line_waves[0] if len(line_waves) <= 2 else line_waves[1]
            # multilet that needs to be fitted together
            if '&' in line:  # Special case for doublet that should be fitted together
                multilet_lines = split_multilet_line(line)
                self.cont_line_v_dict[line] = self.cont_line_dict[line][0] * chosen_line_wave / self.c
                for ii, l in enumerate(multilet_lines):
                    self.model_plus_cont_v_dict[l] = self.model_dict[l] + self.cont_line_v_dict[line]
                    if l in self.broad_wings_lines:
                        self.best_model_plus_cont_n_v_dict[l] = self.best_model_n_dict[l] + self.cont_line_v_dict[line]
                        self.best_model_plus_cont_b_v_dict[l] = self.best_model_b_dict[l] + self.cont_line_v_dict[line]
                        if l in self.triple_gauss_lines:
                            self.best_model_plus_cont_b2_v_dict[l] = self.best_model_b2_dict[l] + self.cont_line_v_dict[line]
                    if l in self.absorption_lines:
                        self.best_model_plus_cont_ab_v_dict[l] = self.best_model_ab_dict[l] + self.cont_line_v_dict[line]
            # single line profile
            else: # General case
                self.cont_line_v_dict[line] = self.cont_line_dict[line][0] * chosen_line_wave / self.c
            self.flux_plus_cont_v_dict[line] = self.flux_v_c_dict[line] # unmasked version (no continuum subtracted)
            self.flux_plus_cont_v_dict_2[line] = self.flux_v_dict[line] + self.cont_line_v_dict[line]
            self.model_plus_cont_v_dict[line] = self.model_dict[line] + self.cont_line_v_dict[line]
            # whether include broad_wings_lines
            if line in self.broad_wings_lines:
                self.best_model_plus_cont_n_v_dict[line] = self.best_model_n_dict[line] + self.cont_line_v_dict[line]
                self.best_model_plus_cont_b_v_dict[line] = self.best_model_b_dict[line] + self.cont_line_v_dict[line]
                if line in self.triple_gauss_lines:
                    self.best_model_plus_cont_b2_v_dict[line] = self.best_model_b2_dict[line] + self.cont_line_v_dict[line]
            # whether include absorption_lines
            if line in self.absorption_lines:
                self.best_model_plus_cont_ab_v_dict[line] = self.best_model_ab_dict[line] + self.cont_line_v_dict[line]

        # plot the fitting results
        for i, line in enumerate(self.selected_lines):
            # Choose the appropriate line wave based on the length of line_waves
            line_waves = self.wave_dict[line]
            chosen_line_wave = line_waves[0] if len(line_waves) <= 2 else line_waves[1]
            if '&' in line:
                multilet_lines = split_multilet_line(line)
                line_name = '&'.join(multilet_lines[:min(len(multilet_lines), 3)])
                v_arr = self.velocity_dict[line][0] if len(multilet_lines) == 2 else self.velocity_dict[line][1]
                v_c_arr = self.velocity_c_dict[line][0] if len(multilet_lines) == 2 else self.velocity_c_dict[line][1]
                for l in multilet_lines:
                    if l in self.broad_wings_lines:
                        axes[0,i].plot(v_arr, self.best_model_plus_cont_n_v_dict[l] / np.max(self.flux_plus_cont_v_dict[line]), 'c--',
                                       zorder = 3, lw = 2)
                        axes[0,i].plot(v_arr, self.best_model_plus_cont_b_v_dict[l] / np.max(self.flux_plus_cont_v_dict[line]), 'b--',
                                       zorder = 3, lw = 2)
                        if l in self.triple_gauss_lines:
                            axes[0,i].plot(v_arr, self.best_model_plus_cont_b2_v_dict[l] / np.max(self.flux_plus_cont_v_dict[line]), 'g--',
                                           zorder = 3, lw = 2)
                    if l in self.absorption_lines:
                        axes[0,i].plot(v_arr, self.best_model_plus_cont_ab_v_dict[l] / np.max(self.flux_plus_cont_v_dict[line]), ls = '--', color = 'orange',
                                       zorder = 3, lw = 2)
            else:
                v_arr = self.velocity_dict[line]
                v_c_arr = self.velocity_c_dict[line]
                line_name = line
            # upper panel for plotting the raw and best-fitting line profile
            axes[0,i].step(v_c_arr, self.flux_plus_cont_v_dict[line]/np.max(self.flux_plus_cont_v_dict[line]), 'k', where = 'mid')
            # axes[0,i].fill_between(v_c_arr, (self.flux_plus_cont_v_dict[line]-self.err_v_c_dict[line]) / np.max(self.flux_plus_cont_v_dict[line]),
            #                       (self.flux_plus_cont_v_dict[line]+self.err_v_c_dict[line]) / np.max(self.flux_plus_cont_v_dict[line]), 
            #                       alpha = 0.5, zorder = 1, facecolor = 'black')
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
                axes[0,i].plot(v_arr, self.best_model_plus_cont_ab_v_dict[line] / np.max(self.flux_plus_cont_v_dict[line]), ls = '--', color = 'orange',
                               zorder = 3, lw = 2)
            axes[0,i].set_yscale('log')
            axes[0,i].text(0.04, 0.92, line_name.replace('&', ' \& ') + '\n' + r'$\chi^2 = $' + "{0:.2f}".format(self.redchi2_dict[line]), 
                           size = 14, transform=axes[0,i].transAxes, va="center",color="black")
            # axes[0,i].axvline(x = 0, ls = '--', color = 'grey', lw = 2) # might be confusing for multiplet
            axes[0,i].tick_params(axis='y', which='minor')
            axes[0,i].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            if i == 0:
                axes[0,i].set_ylabel(r'Normalized Flux',size = 22)
            ymin = 0.9 * min(self.flux_plus_cont_v_dict[line]/np.max(self.flux_plus_cont_v_dict[line]))
            ymax = 1.1
            axes[0,i].set_ylim(ymin, ymax)
            # axes[0,i].legend(loc='upper left', fontsize = 13, framealpha = 0)
            # lower panel for plotting the residual
            axes[1,i].step(v_arr, self.residual_dict[line], 'k', where = 'mid')
            # [axes[1,i].axhline(y=j, color="red", linestyle='--') for j in [0,-1, 1]]
            axes[1,i].set_xlabel(r'Velocity $\rm{(km \ s^{-1})}$',size = 22)
            if i == 0:
                axes[1,i].set_ylabel(r'Residual', size = 22)
            ymin2 = 1.05 * min(self.residual_dict[line])
            ymax2 = 1.05 * max(self.residual_dict[line])
            axes[1,i].set_ylim(ymin2, ymax2)
            # plot the masked regions
            try:
                lmsks = self.lmsks_dict[line]
                if len(lmsks) >= 1:
                    masked_regions = []
                    for ml in lmsks:
                        # define the line masked region in wavelength units
                        x_lmsk = np.linspace(ml['w0'], ml['w1'], 100)
                        # convert from the wavelength units to velocity units
                        v_lmsk = (x_lmsk / chosen_line_wave - 1) * self.c
                        # plot the masked region as fill_between band
                        axes[0,i].fill_between(v_lmsk, ymin, ymax, alpha=0.3, zorder=1, facecolor='orange')
                        axes[1,i].fill_between(v_lmsk, ymin2, ymax2, alpha=0.3, zorder=1, facecolor='orange')
                        
                        # save the bounds of the masked region in velocity units
                        masked_regions.append([v_lmsk[0], v_lmsk[-1]])

                    # Mask and plot regions outside the masked areas
                    errorbar_mask = np.where(v_arr <= masked_regions[0][0])
                    errorbar_mask2 = np.where(v_arr >= masked_regions[-1][-1])

                    # Plot the flux/error masked regions
                    for errorbar_msk in [errorbar_mask, errorbar_mask2]:
                        axes[0,i].fill_between(
                            v_arr[errorbar_msk],
                            (self.flux_plus_cont_v_dict_2[line][errorbar_msk] - self.err_v_dict[line][errorbar_msk]) / np.max(self.flux_plus_cont_v_dict[line]),
                            (self.flux_plus_cont_v_dict_2[line][errorbar_msk] + self.err_v_dict[line][errorbar_msk]) / np.max(self.flux_plus_cont_v_dict[line]),
                            alpha=0.5, zorder=1, facecolor='black'
                        )

                    # Mask and plot the intermediate regions
                    for indx in range(1, len(masked_regions)):
                        errorbar_mask = np.where((v_arr >= masked_regions[indx-1][-1]) & (v_arr <= masked_regions[indx][0]))
                        axes[0,i].fill_between(
                            v_arr[errorbar_mask],
                            (self.flux_plus_cont_v_dict_2[line][errorbar_mask] - self.err_v_dict[line][errorbar_mask]) / np.max(self.flux_plus_cont_v_dict[line]),
                            (self.flux_plus_cont_v_dict_2[line][errorbar_mask] + self.err_v_dict[line][errorbar_mask]) / np.max(self.flux_plus_cont_v_dict[line]),
                            alpha=0.5, zorder=1, facecolor='black'
                        )
            except:
                axes[0,i].fill_between(v_arr, (self.flux_plus_cont_v_dict_2[line]-self.err_v_dict[line]) / np.max(self.flux_plus_cont_v_dict[line]),
                                       (self.flux_plus_cont_v_dict_2[line]+self.err_v_dict[line]) / np.max(self.flux_plus_cont_v_dict[line]), 
                                       alpha = 0.5, zorder = 1, facecolor = 'black')
                print(f"\nno masked regions to plot for {line}.")
        # whether to save the figure
        if savefig:
            # define the current working directory
            current_direc = os.getcwd()
            # define the results directory based on the sub-folder name
            results_dir = os.path.join(current_direc, f"plots/{self.folder_name}/")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            # define the filename and save it in the sub-directory output_files
            filepath = os.path.join(results_dir, self.file_name + '_fittings.pdf')
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.clf() # not showing the matplotlib figure; check the "plots" folder


     