# This line-fitting class is desgined for fitting all lines in velocity space.

import numpy as np
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
from termcolor import colored
from time import time

from veremisfitting.modeling_utils import *
from veremisfitting.analysis_utils import split_multilet_line, extract_key_parts_from_ratio
from IPython import embed

# define timeit to calculate the execution time
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(colored(f"\nTook {end_time - start_time:.4f} seconds to find the best-fitting results.\n", 'green', attrs=['bold', 'underline']))
        return result
    return wrapper

class fitting_model():
    """
    This class is designed for fitting multiple Gaussian and Lorentzian profiles
    to various emission and absorption lines in spectroscopic data.
    The class provides methods to fit individual and combined line profiles.
    """
    def __init__(self, seed = 42):
        """
        Initialize the class by setting the random number generator.
        """
        self.rng = np.random.default_rng(seed) 

    def get_broad_amp(self, dict_pars, num, lines, suffix="", with_amp_in_key = False):
        """Helper function to retrieve the broad amplitude based on the num_comp value."""
        if num == 1:
            return []
        else:
            if not with_amp_in_key:
                return [dict_pars[f"{lines.split(' ')[1]}_b{suffix}"]]
            if with_amp_in_key:
                return [dict_pars[f"amp_{lines.split(' ')[1]}_b{suffix}"]]

    def residual_v_f_all(self, params, x_dict, y_dict, yerr_dict, absorption_lines, broad_wings_lines, double_gauss_lines, triple_gauss_lines):
        """
        Calculate residuals for fitting of all emission lines of interests.

        Args:
            params (dict): Dictionary of fitting parameters.
            x_dict (dict): Input x data dict for each emission line.
            y_dict (dict): Input y data dict for each emission line.
            yerr_dict (dict): Input y error dict for each emission line.
            absorption_lines (list): The list that contains the lines that have absorption troughs.
            broad_wings_lines (list): The list that contains the lines that have broad wings.
            double_gauss_lines (list): The list that contains the lines that need a double-Gaussian model.
            triple_gauss_lines (list): The list that contains the lines that need a triple-Gaussian model.

        Returns:
            residual_all (array): Concatenated array of residuals for each emission line.
        """
        # velocity center and width for the narrow emission component
        x0_e = params['center_e']
        sigma_e = params['sigma_e']
        if broad_wings_lines:
            x0_b = params['center_b']
            sigma_b = params['sigma_b']
            if triple_gauss_lines:
                x0_b2 = params['center_b2']
                sigma_b2 = params['sigma_b2']
        if absorption_lines:
            x0_a = params['center_a']
            sigma_a = params['sigma_a']

        # all intended lines 
        lines = self.emission_lines
        # initialize the residual list
        residuals_all = []

        # iterate over each line
        for line in lines:
            y = y_dict[line]
            yerr = yerr_dict[line]
            # multilet like [OII] 3726&3729
            if '&' in line:
                multilet_lines = split_multilet_line(line)
                # doublet
                if len(multilet_lines) == 2:
                    x, x2 = x_dict[line]
                # triplet
                if len(multilet_lines) == 3:
                    x, x2, x3 = x_dict[line]
                amps = [params['amp_'+key] for key in line.split(' ')[1].split('&')]
                # single emission component
                if all((l not in broad_wings_lines) for l in multilet_lines):
                    params_line = [x0_e, sigma_e] + amps
                    # doublet
                    if len(multilet_lines) == 2:
                        residuals_all.append(residual_2p_v_c_doublet(params_line, x, x2, y, yerr))
                    # triplet
                    if len(multilet_lines) == 3:
                        residuals_all.append(residual_3p_v_c_triplet(params_line, x, x2, x3, y, yerr))
                # multi emission components
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

                    # Initialize the base params
                    params_line = [x0_e, sigma_e] + amps + [x0_b, sigma_b]

                    # Double line profiles
                    if len(multilet_lines) == 2:
                        broad_amp_1 = self.get_broad_amp(params, num_comp_first, multilet_lines[0], with_amp_in_key = True)
                        broad_amp_2 = self.get_broad_amp(params, num_comp_second, multilet_lines[1], with_amp_in_key = True)
                        params_line += broad_amp_1 + broad_amp_2
                        if num_comp_first == 3 or num_comp_second == 3:
                            params_line += [x0_b2, sigma_b2] 
                            if num_comp_first == 3:
                                params_line += self.get_broad_amp(params, num_comp_first, multilet_lines[0], "2", with_amp_in_key = True) 
                            if num_comp_second == 3:
                                params_line += self.get_broad_amp(params, num_comp_second, multilet_lines[1], "2", with_amp_in_key = True)
                        # append the line residual to the residual dict
                        # whether use the lorentzian function to fit broad wings or not
                        if ((multilet_lines[0] in self.fit_func_choices.keys()) and self.fit_func_choices[multilet_lines[0]] == 'Lorentzian') or \
                           ((multilet_lines[1] in self.fit_func_choices.keys()) and self.fit_func_choices[multilet_lines[1]] == 'Lorentzian'):
                            residuals_all.append(residual_2p_gl_v_c_doublet(params_line, x, x2, y, yerr, num_comp_first=num_comp_first, num_comp_second=num_comp_second))

                        else:
                            residuals_all.append(residual_2p_v_c_doublet(params_line, x, x2, y, yerr, num_comp_first=num_comp_first, num_comp_second=num_comp_second))

                    # Triple line profiles
                    if len(multilet_lines) == 3:
                        broad_amp_1 = self.get_broad_amp(params, num_comp_first, multilet_lines[0], with_amp_in_key = True)
                        broad_amp_2 = self.get_broad_amp(params, num_comp_second, multilet_lines[1], with_amp_in_key = True)
                        broad_amp_3 = self.get_broad_amp(params, num_comp_third, multilet_lines[2], with_amp_in_key = True)
                        params_line += broad_amp_1 + broad_amp_2 + broad_amp_3
                        if any(x == 3 for x in [num_comp_first, num_comp_second, num_comp_third]):
                            params_line += [x0_b2, sigma_b2] 
                            if num_comp_first == 3:
                                params_line += self.get_broad_amp(params, num_comp_first, multilet_lines[0], "2", with_amp_in_key = True) 
                            if num_comp_second == 3:
                                params_line += self.get_broad_amp(params, num_comp_second, multilet_lines[1], "2", with_amp_in_key = True)
                            if num_comp_third == 3:
                                params_line += self.get_broad_amp(params, num_comp_third, multilet_lines[2], "2", with_amp_in_key = True)
                        # append the line residual to the residual dict
                        # whether use the lorentzian function to fit broad wings or not
                        if ((multilet_lines[0] in self.fit_func_choices.keys()) and self.fit_func_choices[multilet_lines[0]] == 'Lorentzian') or \
                           ((multilet_lines[1] in self.fit_func_choices.keys()) and self.fit_func_choices[multilet_lines[1]] == 'Lorentzian') or \
                           ((multilet_lines[2] in self.fit_func_choices.keys()) and self.fit_func_choices[multilet_lines[2]] == 'Lorentzian'):
                            residuals_all.append(residual_3p_gl_v_c_triplet(params_line, x, x2, x3, y, yerr, 
                                                 num_comp_first=num_comp_first, num_comp_second=num_comp_second, num_comp_third=num_comp_third))
                        else:
                            residuals_all.append(residual_3p_v_c_triplet(params_line, x, x2, x3, y, yerr, 
                                                 num_comp_first=num_comp_first, num_comp_second=num_comp_second, num_comp_third=num_comp_third))

            # single line profile
            else:
                x = x_dict[line]
                amps = [params[f"amp_{line.split(' ')[1]}"]]
                # single emission component
                if (line not in broad_wings_lines) and (line not in absorption_lines):
                    params_line = [x0_e, sigma_e] + amps
                    residuals_all.append(residual_1p_v_c(params_line, x, y, yerr))
                # multi emission components
                if (line in broad_wings_lines) and (line not in absorption_lines):
                    # double emission components
                    if line in double_gauss_lines:
                        broad_amps = [params[f"amp_{line.split(' ')[1]}_b"]]
                        params_line = [x0_e, x0_b, sigma_e, sigma_b] + amps + broad_amps
                        if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian'):
                            residuals_all.append(residual_2p_gl_v_c(params_line, x, y, yerr))
                        else:
                            residuals_all.append(residual_2p_v_c(params_line, x, y, yerr))
                    # triple emission components
                    if line in triple_gauss_lines:
                        broad_amps = [params[f"amp_{line.split(' ')[1]}_b"]] + [params[f"amp_{line.split(' ')[1]}_b2"]]
                        params_line = [x0_e, x0_b, x0_b2, sigma_e, sigma_b, sigma_b2] + amps + broad_amps
                        if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian'):
                            residuals_all.append(residual_3p_gl_v_c(params_line, x, y, yerr))
                        else:
                            residuals_all.append(residual_3p_v_c(params_line, x, y, yerr))
                # emission + absorption components
                if (line in absorption_lines):
                    abs_amps = [params[f"amp_{line.split(' ')[1]}_abs"]]
                    if line not in broad_wings_lines:
                        params_line = [x0_e, x0_a, sigma_e, sigma_a] + amps + abs_amps
                        if (line in self.fit_func_abs_choices.keys()) and (self.fit_func_abs_choices[line] == 'Lorentzian'):
                            residuals_all.append(residual_2p_gl_v_c(params_line, x, y, yerr))
                        else:
                            residuals_all.append(residual_2p_v_c(params_line, x, y, yerr))
                    else:
                        if line in double_gauss_lines:
                            broad_amps = [params[f"amp_{line.split(' ')[1]}_b"]]
                            params_line = [x0_e, x0_b, x0_a, sigma_e, sigma_b, sigma_a] + amps + broad_amps + abs_amps
                            if (line in self.fit_func_abs_choices.keys()) and (self.fit_func_abs_choices[line] == 'Lorentzian'):
                                residuals_all.append(residual_3p_gl_v_c(params_line, x, y, yerr))
                            else:
                                residuals_all.append(residual_3p_v_c(params_line, x, y, yerr))
                        elif line in triple_gauss_lines:
                            broad_amps = [params[f"amp_{line.split(' ')[1]}_b"]] + [params[f"amp_{line.split(' ')[1]}_b2"]]
                            params_line = [x0_e, x0_b, x0_b2, x0_a, sigma_e, sigma_b, sigma_b2, sigma_a] + amps + broad_amps + abs_amps
                            if (line in self.fit_func_abs_choices.keys()) and (self.fit_func_abs_choices[line] == 'Lorentzian'):
                                residuals_all.append(residual_4p_gl_v_c(params_line, x, y, yerr))
                            else:
                                residuals_all.append(residual_4p_v_c(params_line, x, y, yerr))

        return np.concatenate(residuals_all)

    def print_best_fit_params(self, x0_e, sigma_e, amps, x0_b=None, sigma_b=None, x0_b2=None, sigma_b2=None, x0_a=None, sigma_a=None, 
                              broad_wing=False, absorption=False, double_gauss=False, triple_gauss=False):
        # Printing velocity centers
        velocity_centers = colored(f"velocity center (in km/s):", 'grey', attrs=['underline']) + f" x0_e = {x0_e:.3f}"
        if broad_wing:
            velocity_centers += f", x0_b = {x0_b:.3f}"
            if triple_gauss:
                velocity_centers += f", x0_b2 = {x0_b2:.3f}"
        if absorption:
            velocity_centers += f", x0_a = {x0_a:.3f}"
        print(velocity_centers)
        
        # Printing velocity widths
        velocity_widths = colored(f"velocity width (in km/s):", 'grey', attrs=['underline']) + f" sigma_e = {sigma_e:.3f}"
        if broad_wing:
            velocity_widths += f", sigma_b = {sigma_b:.3f}"
            if triple_gauss:
                velocity_widths += f", sigma_b2 = {sigma_b2:.3f}"
        if absorption:
            velocity_widths += f", sigma_a = {sigma_a:.3f}"
        print(velocity_widths)
        
        # Printing line amplitudes
        print(colored(r"line amplitude (in Flam units):", 'grey', attrs=['underline']))
        for line in self.emission_lines:
            self.print_line_amplitude(line, amps, broad_wing, absorption, double_gauss, triple_gauss)

    def print_line_amplitude(self, line, amps, broad_wing, absorption, double_gauss, triple_gauss):
        line_parts = line.split(' ')[1]
        components = line_parts.split('&') if '&' in line_parts else [line_parts]
        
        amplitude_values = []
        for component in components:
            base_amp = amps.get(component, 0)
            amplitude_values.append(f"{base_amp:.3f}")
            
            if broad_wing:
                if double_gauss or triple_gauss:
                    amp_b = amps.get(f"{component}_b", 0)
                    if amp_b:
                        amplitude_values.append(f"{amp_b:.3f}")
                if triple_gauss:
                    amp_b2 = amps.get(f"{component}_b2", 0)
                    if amp_b2:
                        amplitude_values.append(f"{amp_b2:.3f}")
            if absorption:
                amp_abs = amps.get(f"{component}_abs", 0)
                if amp_abs:
                    amplitude_values.append(f"{amp_abs:.3f}")
        
        # Joining all amplitude values into a single string for the line
        amplitude_str = ", ".join(amplitude_values)
        print(f"{line_parts}: [{amplitude_str}]")

    def assign_best(self, x0_e, sigma_e, amps, x0_b=None, sigma_b=None, x0_b2=None, sigma_b2=None, x0_a=None, sigma_a=None, 
                    broad_wing=False, absorption=False, double_gauss=False, triple_gauss=False):
        """
        Assign the best parameters, result, and model.

        Parameters
        ----------
        model_dict : dict
            Dictionary of the model components
        x0_e, sigma_e : float
            Parameters for the emission line
        amps : dict
            Dictionary of amplitudes for each line
        x0_b, sigma_b : float, optional
            Parameters for the second emission component
        x0_b2, sigma_b2 : float, optional
            Parameters for the third emission component
        x0_a, sigma_a : float, optional
            Parameters for the absorption component
        broad_wing : bool
            Whether the multi emission component exists
        absorption : bool
            Whether the absorption component exists
        double_gauss : bool
            Whether the broad-wing lines need a double gaussian model
        triple_gauss : bool
            Whether the broad-wing lines need a triple gaussian model
        """
        self.best_params = np.array([x0_e, sigma_e] + [amp for line in self.emission_lines \
                                     for amp in ([amps[key] for key in line.split(' ')[1].split('&')] if '&' in line else [amps[line.split(' ')[1]]])])

        if broad_wing and (not absorption):
            if (not triple_gauss):
                self.best_params = np.append(self.best_params, [x0_b, sigma_b] + [amps[line.split(' ')[1]+'_b'] for line in self.double_gauss_lines])
            if triple_gauss:
                self.best_params = np.append(self.best_params, [x0_b, sigma_b] + [amps[line.split(' ')[1]+'_b'] for line in self.broad_wings_lines])
                self.best_params = np.append(self.best_params, [x0_b2, sigma_b2] + [amps[line.split(' ')[1]+'_b2'] for line in self.triple_gauss_lines])
        if absorption and (not broad_wing):
            self.best_params = np.append(self.best_params, [x0_a, sigma_a] + [amps[line.split(' ')[1]+'_abs'] for line in self.absorption_lines])
        if absorption and broad_wing:
            self.best_params = np.append(self.best_params, [x0_a, sigma_a] + [amps[line.split(' ')[1]+'_abs'] for line in self.absorption_lines])
            if (not triple_gauss):
                self.best_params = np.append(self.best_params, [x0_b, sigma_b] + [amps[line.split(' ')[1]+'_b'] for line in self.double_gauss_lines])
            if triple_gauss:
                self.best_params = np.append(self.best_params, [x0_b, sigma_b] + [amps[line.split(' ')[1]+'_b'] for line in self.broad_wings_lines])
                self.best_params = np.append(self.best_params, [x0_b2, sigma_b2] + [amps[line.split(' ')[1]+'_b2'] for line in self.triple_gauss_lines])
        
        # print the best chi2 and best parameter values of the current iteration (which satisfies the conditions above)
        print(colored(f"\nIteration #{self.current_iteration}: ", 'green', attrs=['bold', 'underline']))
        print(colored("The current best chi2 value is ", 'green'))
        print(colored("{0:.5f}".format(self.best_chi2), 'grey'))
        print(colored("The current best parameter values are ", 'green'))
        self.print_best_fit_params(x0_e, sigma_e, amps, x0_b=x0_b, sigma_b=sigma_b, x0_b2=x0_b2, sigma_b2=sigma_b2, x0_a=x0_a, sigma_a=sigma_a, 
                                   broad_wing=broad_wing, absorption=absorption, double_gauss=double_gauss, triple_gauss=triple_gauss)

    def check_and_assign_best(self, amps, absorption=False, broad_wing=False, double_gauss=False, triple_gauss=False):
        """
        Check the model fitting conditions and assign the best parameters, result, and model if they meet the conditions.

        Parameters
        ----------
        model_dict : dict
            Dictionary of the model components
        amps : dict
            Dictionary of amplitudes for each line
        broad_wing : bool
            Whether the broad wing component exists
        absorption : bool
            Whether the absorption component exists
        double_gauss : bool
            Whether the broad-wing lines need a double gaussian model
        triple_gauss : bool
            Whether the broad-wing lines need a triple gaussian model
        """
        # best-fitting lmfit result
        self.best_res = self.result
        # best-fitting reduced chi2
        self.best_chi2 = self.result.redchi
        # best-fitting amps dictionary
        self.best_amps = self.amps
        # best-fitting param dict
        self.best_param_dict = self.param_dict
        # obtain the best-fitting velocity center and width of each component
        self.x0_e = self.best_param_dict["center_e"]
        self.sigma_e = self.best_param_dict["sigma_e"]
        self.x0_a = self.best_param_dict.get("center_a", None)
        self.sigma_a = self.best_param_dict.get("sigma_a", None)
        self.x0_b = self.best_param_dict.get("center_b", None)
        self.sigma_b = self.best_param_dict.get("sigma_b", None)
        self.x0_b2 = self.best_param_dict.get("center_b2", None)
        self.sigma_b2 = self.best_param_dict.get("sigma_b2", None)

        if (not absorption) and (not broad_wing):
            self.assign_best(self.x0_e, self.sigma_e, amps)
        elif (not absorption) and broad_wing:
            self.assign_best(self.x0_e, self.sigma_e, amps, x0_b=self.x0_b, sigma_b=self.sigma_b, x0_b2=self.x0_b2, sigma_b2=self.sigma_b2, broad_wing=True, 
                             double_gauss=double_gauss, triple_gauss=triple_gauss)
        elif absorption and (not broad_wing):
            self.assign_best(self.x0_e, self.sigma_e, amps, x0_a=self.x0_a, sigma_a=self.sigma_a, absorption=True)
        elif absorption and broad_wing:
            self.assign_best(self.x0_e, self.sigma_e, amps, x0_a=self.x0_a, sigma_a=self.sigma_a, x0_b=self.x0_b, sigma_b=self.sigma_b, x0_b2=self.x0_b2, sigma_b2=self.sigma_b2,
                             absorption=True, broad_wing=True, double_gauss=double_gauss, triple_gauss=triple_gauss)

    @timeit
    def fitting_all_lines(self, input_arr, n_iteration = 1000):
        """
        Fit all intended emission lines in the velocity space using a specified number of iterations.

        Parameters
        ----------
        input_arr : list
            A list containing, e.g., velocity_arr, flux_v_arr, err_v_arr, initial_guess, and param_range.
        n_iteration : int, optional
            Number of iterations for fitting, default is 1000.
        Returns
        -------
        best_model (array): The best fitting model found after iterating through the fitting process.
        best_chi2 (float): The reduced chi-squared value corresponding to the best fitting model.
        """
        # fit all intended emission lines in the velocity space
        velocity_dict, flux_v_dict, err_v_dict, initial_guess_dict, param_range_dict, amps_ratio_dict, self.absorption_lines, self.broad_wings_lines, self.double_gauss_lines, \
        self.triple_gauss_lines, self.double_gauss_broad, self.triple_gauss_broad, fitting_method, self.fit_func_choices, self.fit_func_abs_choices, self.sigma_limits, self.fit_algorithm = input_arr

        # get a copy of the initial guess dict
        initial_guess_dict_old = initial_guess_dict.copy()

        # all intended emission lines
        self.emission_lines = list(velocity_dict.keys())

        # define the lists that contains the fixed amplitude pairs and their fixed amp ratios
        self.amps_fixed_list = [part for ratio, value in amps_ratio_dict.items() for part in extract_key_parts_from_ratio(ratio)]
        self.amps_ratio_list = [value for ratio, value in amps_ratio_dict.items()]

        # determine whether to fix the velocity center and width for multi-component fittings
        fitting_methods = {'Free fitting': (True, True), 'Fix velocity centroid': (False, True), 'Fix velocity width': (True, False), 
                           'Fix velocity centroid and width': (False, False)}
        vary_center, vary_width = fitting_methods[fitting_method]
        vary_dict = {'e': (True, True), 'b': (vary_center, vary_width), 'b2': (vary_center, vary_width),  'a': (vary_center, vary_width)}

        # assign the input limits of velocity width of each velocity component
        sigma_min, sigma_max_e, sigma_max_a = self.sigma_limits

        for i in range(n_iteration):
            # define the current iteration number
            self.current_iteration = i + 1
            # define the input parameters
            self.params = Parameters()
            # velocity center and width of the narrow emission component
            self.params.add('center_e', value=initial_guess_dict['v_e'][0], vary = True, min = -200, max = 200)
            self.params.add('sigma_e', value=initial_guess_dict['v_e'][1], min = sigma_min, max = sigma_max_e, vary = True)
            for component in ['b', 'b2', 'a']:
                # define a max sigma value 
                sigma_max = sigma_max_a if component == 'a' else sigma_max_e
                # set initial values only for other velocity info
                if f'v_{component}' in initial_guess_dict:
                    self.params.add(f"center_{component}", value=initial_guess_dict[f'v_{component}'][0], min = -200, max = 200,
                                    expr=f"center_e" if not vary_dict[component][0] else None)

                    if self.double_gauss_broad and (component == 'b'): # check there are any broad velocity components for multi-component models
                        self.params.add(f"sigma_delta_b", value = initial_guess_dict[f'v_{component}'][1] - initial_guess_dict['v_e'][1], min = 0) 
                        self.params.add('sigma_b', expr='sigma_e + sigma_delta_b', min=sigma_min, max=sigma_max)

                    elif self.triple_gauss_broad and (component == 'b2'): # check there are any broad velocity components for triple-component models 
                        self.params.add(f"sigma_delta_b2", value = initial_guess_dict[f'v_{component}'][1] - initial_guess_dict['v_b'][1], min = 0) 
                        self.params.add('sigma_b2', expr='sigma_b + sigma_delta_b2', min=sigma_min, max=sigma_max)

                    elif component == 'a': # the absorption component's velocity width should be always larger than that of the narrow component
                        self.params.add(f"sigma_delta_a", value = initial_guess_dict[f'v_{component}'][1] - initial_guess_dict['v_e'][1], min = 0)
                        self.params.add('sigma_a', expr='sigma_e + sigma_delta_a', min=sigma_min, max=sigma_max)

                    else: # use the default constraints sigma_min and sigma_max
                        self.params.add(f"sigma_{component}", value=initial_guess_dict[f'v_{component}'][1], min=sigma_min, max=sigma_max, 
                                        expr=f"sigma_e" if not vary_dict[component][1] else None)

            # set initial values only for amplitude info (free amplitude fitting)
            for line, initial_guess in initial_guess_dict.items():
                ion_wave_split = line.split(' ')
                if len(ion_wave_split) > 1: 
                    # lines that follow fixed amp ratio fitting strategy
                    if any(f"{ion_wave_split[1]}{suffix}" in self.amps_fixed_list for suffix in ["", "_b", "_b2", "_abs"]):
                        if ion_wave_split[1] in self.amps_fixed_list:
                            indx_num = self.amps_fixed_list.index(ion_wave_split[1])
                            if indx_num % 2 != 0:
                                amp_ratio_indx = int((indx_num + 1) / 2 - 1)
                                amp_ratio = self.amps_ratio_list[indx_num - 1] # fixed amp ratio between these two lines
                                self.params.add(f"amp_{ion_wave_split[1]}", value=initial_guess[0], min = 0) # first line 
                                self.params.add(f"amp_{self.amps_fixed_list[indx_num - 1]}", expr = f"{amp_ratio} * amp_{ion_wave_split[1]}") # second line

                        if (line in self.broad_wings_lines):
                            # check whether the lines' second velocity component has fixed ratio or not
                            if f"{ion_wave_split[1]}_b" in self.amps_fixed_list:
                                indx_num_b = self.amps_fixed_list.index(f"{ion_wave_split[1]}_b")
                                if indx_num_b % 2 != 0:
                                    amp_ratio_b_indx = int((indx_num_b + 1) / 2 - 1)
                                    amp_ratio_b = self.amps_ratio_list[amp_ratio_b_indx] # fixed amp ratio between these two lines
                                    self.params.add(f"amp_{ion_wave_split[1]}_b", value=initial_guess[1], min = 0) # first line 
                                    self.params.add(f"amp_{self.amps_fixed_list[indx_num_b - 1]}", expr=f"{amp_ratio_b} * amp_{ion_wave_split[1]}_b") # second line
                            else:
                                self.params.add(f"amp_{ion_wave_split[1]}_b", value=initial_guess[1], min = 0)
                                
                        if (line in self.triple_gauss_lines):
                            # check whether the lines' third velocity component has fixed ratio or not
                            if f"{ion_wave_split[1]}_b2" in self.amps_fixed_list:
                                indx_num_b2 = self.amps_fixed_list.index(f"{ion_wave_split[1]}_b2")
                                if indx_num_b2 % 2 != 0:
                                    amp_ratio_b2_indx = int((indx_num_b2 + 1) / 2 - 1)
                                    amp_ratio_b2 = self.amps_ratio_list[amp_ratio_b2_indx] # fixed amp ratio between these two lines
                                    self.params.add(f"amp_{ion_wave_split[1]}_b2", value=initial_guess[2], min = 0) # first line 
                                    self.params.add(f"amp_{self.amps_fixed_list[indx_num_b2 - 1]}", expr = f"{amp_ratio_b2} * amp_{ion_wave_split[1]}_b2") # second line
                            else:
                                self.params.add(f"amp_{ion_wave_split[1]}_b2", value=initial_guess[2], min = 0)

                        if (line in self.absorption_lines):
                            # check whether the lines' absorption component has fixed ratio or not
                            if f"{ion_wave_split[1]}_abs" in self.amps_fixed_list:
                                indx_num_abs = self.amps_fixed_list.index(f"{ion_wave_split[1]}_abs")
                                if indx_num_abs % 2 != 0:
                                    amp_ratio_abs_indx = int((indx_num_abs + 1) / 2 - 1)
                                    amp_ratio_abs = self.amps_ratio_list[amp_ratio_abs_indx] # fixed amp ratio between these two lines
                                    self.params.add(f"amp_{ion_wave_split[1]}_abs", value=initial_guess[-1], max = 0) # first line 
                                    self.params.add(f"amp_{self.amps_fixed_list[indx_num_abs - 1]}", expr = f"{amp_ratio_abs} * amp_{ion_wave_split[1]}_abs") # second line
                            else:
                                self.params.add(f"amp_{ion_wave_split[1]}_abs", value=initial_guess[-1], max = 0)

                    # lines that follow free amplitude fitting strategy
                    if any(f"{ion_wave_split[1]}{suffix}" not in self.amps_fixed_list for suffix in ["", "_b", "_b2", "_abs"]):
                        if ion_wave_split[1] not in self.amps_fixed_list:
                            self.params.add(f"amp_{ion_wave_split[1]}", value=initial_guess[0], min = 0)
                        if (line in self.broad_wings_lines) and (f"{ion_wave_split[1]}_b" not in self.amps_fixed_list):
                            self.params.add(f"amp_{ion_wave_split[1]}_b", value=initial_guess[1], min = 0)
                            if (line in self.triple_gauss_lines) and (f"{ion_wave_split[1]}_b2" not in self.amps_fixed_list):
                                self.params.add(f"amp_{ion_wave_split[1]}_b2", value=initial_guess[2], min = 0)
                        if (line in self.absorption_lines) and (f"{ion_wave_split[1]}_abs" not in self.amps_fixed_list):
                            self.params.add(f"amp_{ion_wave_split[1]}_abs", value=initial_guess[-1], max = 0)

            # obtain the best result of this iteration
            self.result = minimize(self.residual_v_f_all, self.params, args=(velocity_dict, flux_v_dict, err_v_dict, self.absorption_lines, self.broad_wings_lines, 
                                                                             self.double_gauss_lines, self.triple_gauss_lines), method = self.fit_algorithm, calc_covar = True, 
                                                                             max_nfev=100000)
            self.param_dict = self.result.params.valuesdict()

            # collect amplitude values
            self.amps = {key.replace('amp_', ''): value for key, value in self.param_dict.items() if 'amp' in key}
               
            try:
                if (self.best_chi2 > self.result.redchi):
                    # print and assign the best-fitting parameters (and also print the best chi2 value)
                    self.check_and_assign_best(self.amps, absorption=bool(self.absorption_lines), broad_wing=bool(self.broad_wings_lines),
                                               double_gauss=bool(self.double_gauss_lines), triple_gauss=bool(self.triple_gauss_lines))
            except (UnboundLocalError, NameError, AttributeError) as e:
                # print and assign the best-fitting parameters (and also print the best chi2 value)
                self.check_and_assign_best(self.amps, absorption=bool(self.absorption_lines), broad_wing=bool(self.broad_wings_lines),
                                           double_gauss=bool(self.double_gauss_lines), triple_gauss=bool(self.triple_gauss_lines))
            # update the initial guess dict based on the parameter range dict
            initial_guess_dict = {key: value + np.float64((2*self.rng.random(1)-1))*param_range_dict[key] for key, value in initial_guess_dict_old.items()}

        ################# Start: obtain the best-fitting model for each line #################
        # initialize dicts for saving models for multiple emission components
        self.best_model_n_dict = dict()
        if self.broad_wings_lines:
            self.best_model_b_dict = dict()
            if self.triple_gauss_lines:
                self.best_model_b2_dict = dict()
                
        # initialize dicts for saving models for emission and absorption components
        if self.absorption_lines:
            self.best_model_ab_dict = dict()

        # model for each component
        self.best_model = dict()
        self.residual_dict = dict()
        self.params_num_dict = dict()
        for line in velocity_dict.keys():
            # doublet that needs to be fitted together
            if '&' in line:
                multilet_lines = split_multilet_line(line)
                amps = [self.best_amps[l.split(' ')[1]] for l in multilet_lines]
                absorption_check = np.array([l in self.absorption_lines for l in multilet_lines])
                # single emission component
                if all((l not in self.broad_wings_lines) for l in multilet_lines):
                    params_line = [self.x0_e, self.sigma_e] + amps
                    if any(absorption_check):
                        params_line += [self.x0_a, self.sigma_a] + [self.best_amps[l.split(' ')[1] + '_abs'] for l in multilet_lines if l in self.absorption_lines]
                    # doublet
                    if len(multilet_lines) == 2:
                        comp_first_abs, comp_second_abs = np.array([True] * 2) * absorption_check
                        abs_lorentz = True if ((multilet_lines[0] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[0]] == 'Lorentzian') or \
                                               (multilet_lines[1] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[1]] == 'Lorentzian')) else False
                        # best model
                        self.best_model[multilet_lines[0]], self.best_model[multilet_lines[1]], self.best_model[line] = \
                                gaussian_2p_v_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                      comp_first_abs = comp_first_abs, comp_second_abs = comp_second_abs, abs_lorentz = abs_lorentz)
                        # residual
                        self.residual_dict[line] = residual_2p_v_c_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                                           flux_v_dict[line], err_v_dict[line], comp_first_abs = comp_first_abs, 
                                                                           comp_second_abs = comp_second_abs, abs_lorentz = abs_lorentz)
                        if comp_first_abs:
                            abs_amp = [self.best_amps[f"{multilet_lines[0].split(' ')[1]}_abs"]]
                            if (multilet_lines[0] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[0]] == 'Lorentzian'):
                                self.best_model_ab_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model_ab_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], self.x0_a, self.sigma_a, abs_amp[0])
                        if comp_second_abs:
                            abs_amp = [self.best_amps[f"{multilet_lines[1].split(' ')[1]}_abs"]]
                            if (multilet_lines[1] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[1]] == 'Lorentzian'):
                                self.best_model_ab_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model_ab_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], self.x0_a, self.sigma_a, abs_amp[0])
                    # triplet
                    elif len(multilet_lines) == 3:
                        comp_first_abs, comp_second_abs, comp_third_abs = np.array([True] * 3) * absorption_check
                        abs_lorentz = True if ((multilet_lines[0] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[0]] == 'Lorentzian') or \
                                               (multilet_lines[1] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[1]] == 'Lorentzian') or \
                                               (multilet_lines[2] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[1]] == 'Lorentzian')) else False
                        self.best_model[multilet_lines[0]], self.best_model[multilet_lines[1]], self.best_model[multilet_lines[2]], self.best_model[line] = \
                                gaussian_3p_v_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], velocity_dict[line][2], 
                                                      comp_first_abs = comp_first_abs, comp_second_abs = comp_second_abs, comp_third_abs = comp_third_abs, abs_lorentz = abs_lorentz)
                        # residual
                        self.residual_dict[line] = residual_3p_v_c_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], velocity_dict[line][2],
                                                                           flux_v_dict[line], err_v_dict[line], comp_first_abs = comp_first_abs, comp_second_abs = comp_second_abs, 
                                                                           comp_third_abs = comp_third_abs, abs_lorentz = abs_lorentz)
                        if comp_first_abs:
                            abs_amp = [self.best_amps[f"{multilet_lines[0].split(' ')[1]}_abs"]]
                            if (multilet_lines[0] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[0]] == 'Lorentzian'):
                                self.best_model_ab_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model_ab_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], self.x0_a, self.sigma_a, abs_amp[0])
                        if comp_second_abs:
                            abs_amp = [self.best_amps[f"{multilet_lines[1].split(' ')[1]}_abs"]]
                            if (multilet_lines[1] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[1]] == 'Lorentzian'):
                                self.best_model_ab_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model_ab_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], self.x0_a, self.sigma_a, abs_amp[0])
                        if comp_third_abs:
                            abs_amp = [self.best_amps[f"{multilet_lines[2].split(' ')[1]}_abs"]]
                            if (multilet_lines[2] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[2]] == 'Lorentzian'):
                                self.best_model_ab_dict[multilet_lines[2]] = lorentzian_1p_v(velocity_dict[line][2], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model_ab_dict[multilet_lines[2]] = gaussian_1p_v(velocity_dict[line][2], self.x0_a, self.sigma_a, abs_amp[0])
                # multi emission components
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
                    # define the base params 
                    params_line = [self.x0_e, self.sigma_e] + amps + [self.x0_b, self.sigma_b]
                    # Double line profiles
                    if len(multilet_lines) == 2:
                        broad_amp_1 = self.get_broad_amp(self.best_amps, num_comp_first, multilet_lines[0])
                        self.best_model_n_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], self.x0_e, self.sigma_e, amps[0])
                        if broad_amp_1:
                            if (multilet_lines[0] in self.fit_func_choices.keys()) and (self.fit_func_choices[multilet_lines[0]] == 'Lorentzian') and (max_comp == 2):
                                self.best_model_b_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], self.x0_b, self.sigma_b, broad_amp_1[0])
                            else:
                                self.best_model_b_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], self.x0_b, self.sigma_b, broad_amp_1[0])
                        broad_amp_2 = self.get_broad_amp(self.best_amps, num_comp_second, multilet_lines[1])
                        self.best_model_n_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], self.x0_e, self.sigma_e, amps[1])
                        if broad_amp_2:
                            if (multilet_lines[1] in self.fit_func_choices.keys()) and (self.fit_func_choices[multilet_lines[1]] == 'Lorentzian') and (max_comp == 2):
                                self.best_model_b_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], self.x0_b, self.sigma_b, broad_amp_2[0])
                            else:
                                self.best_model_b_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], self.x0_b, self.sigma_b, broad_amp_2[0])
                        params_line += broad_amp_1 + broad_amp_2
                        if num_comp_first == 3 or num_comp_second == 3:
                            params_line += [self.x0_b2, self.sigma_b2] 
                            if num_comp_first == 3:
                                broad_amp2_1 = self.get_broad_amp(self.best_amps, num_comp_first, multilet_lines[0], "2")
                                params_line += broad_amp2_1
                                if broad_amp2_1:
                                    if (multilet_lines[0] in self.fit_func_choices.keys()) and (self.fit_func_choices[multilet_lines[0]] == 'Lorentzian'):
                                        self.best_model_b2_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], self.x0_b2, self.sigma_b2, broad_amp2_1[0])
                                    else:
                                        self.best_model_b2_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], self.x0_b2, self.sigma_b2, broad_amp2_1[0])
                            if num_comp_second == 3:
                                broad_amp2_2 = self.get_broad_amp(self.best_amps, num_comp_second, multilet_lines[1], "2")
                                params_line += broad_amp2_2
                                if broad_amp2_2:
                                    if (multilet_lines[1] in self.fit_func_choices.keys()) and (self.fit_func_choices[multilet_lines[1]] == 'Lorentzian'):
                                        self.best_model_b2_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], self.x0_b2, self.sigma_b2, broad_amp2_2[0])
                                    else:
                                        self.best_model_b2_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], self.x0_b2, self.sigma_b2, broad_amp2_2[0])

                        # check whether any peak needs an absorption component
                        if any(absorption_check):
                            params_line += [self.x0_a, self.sigma_a] + [self.best_amps[l.split(' ')[1] + '_abs'] for l in multilet_lines if l in self.absorption_lines]
                        comp_first_abs, comp_second_abs = np.array([True] * 2) * absorption_check
                        if comp_first_abs:
                            abs_amp = [self.best_amps[f"{multilet_lines[0].split(' ')[1]}_abs"]]
                            if (multilet_lines[0] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[0]] == 'Lorentzian'):
                                self.best_model_ab_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model_ab_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], self.x0_a, self.sigma_a, abs_amp[0])
                        if comp_second_abs:
                            abs_amp = [self.best_amps[f"{multilet_lines[1].split(' ')[1]}_abs"]]
                            if (multilet_lines[1] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[1]] == 'Lorentzian'):
                                self.best_model_ab_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model_ab_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], self.x0_a, self.sigma_a, abs_amp[0])
                        abs_lorentz = True if ((multilet_lines[0] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[0]] == 'Lorentzian') or \
                                               (multilet_lines[1] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[1]] == 'Lorentzian')) else False

                        # append the line model to the model dict
                        # whether use the lorentzian function to fit broad wings or not
                        if ((multilet_lines[0] in self.fit_func_choices.keys()) and self.fit_func_choices[multilet_lines[0]] == 'Lorentzian') or \
                           ((multilet_lines[1] in self.fit_func_choices.keys()) and self.fit_func_choices[multilet_lines[1]] == 'Lorentzian'):
                            # best model
                            self.best_model[multilet_lines[0]], self.best_model[multilet_lines[1]], self.best_model[line] = \
                            gaussian_lorentzian_2p_v_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], num_comp_first, num_comp_second, comp_first_abs, comp_second_abs, abs_lorentz)
                            # residual
                            self.residual_dict[line] = residual_2p_gl_v_c_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                                                  flux_v_dict[line], err_v_dict[line], 
                                                                                  num_comp_first, num_comp_second, 
                                                                                  comp_first_abs, comp_second_abs, abs_lorentz)
                        else:
                            # best model
                            self.best_model[multilet_lines[0]], self.best_model[multilet_lines[1]], self.best_model[line] = \
                            gaussian_2p_v_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], num_comp_first, num_comp_second, comp_first_abs, comp_second_abs, abs_lorentz)
                            # residual
                            self.residual_dict[line] = residual_2p_v_c_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                                               flux_v_dict[line], err_v_dict[line], 
                                                                               num_comp_first, num_comp_second, 
                                                                               comp_first_abs, comp_second_abs, abs_lorentz)

                    # Triple line profiles
                    if len(multilet_lines) == 3:
                        # line 1
                        broad_amp_1 = self.get_broad_amp(self.best_amps, num_comp_first, multilet_lines[0])
                        self.best_model_n_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], self.x0_e, self.sigma_e, amps[0])
                        if broad_amp_1:
                            if (multilet_lines[0] in self.fit_func_choices.keys()) and (self.fit_func_choices[multilet_lines[0]] == 'Lorentzian') and (max_comp == 2):
                                self.best_model_b_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], self.x0_b, self.sigma_b, broad_amp_1[0])
                            else:
                                self.best_model_b_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], self.x0_b, self.sigma_b, broad_amp_1[0])
                        # line 2
                        broad_amp_2 = self.get_broad_amp(self.best_amps, num_comp_second, multilet_lines[1])
                        self.best_model_n_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], self.x0_e, self.sigma_e, amps[1])
                        if broad_amp_2:
                            if (multilet_lines[1] in self.fit_func_choices.keys()) and (self.fit_func_choices[multilet_lines[1]] == 'Lorentzian') and (max_comp == 2):
                                self.best_model_b_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], self.x0_b, self.sigma_b, broad_amp_2[0])
                            else:
                                self.best_model_b_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], self.x0_b, self.sigma_b, broad_amp_2[0])
                        # line 3
                        broad_amp_3 = self.get_broad_amp(self.best_amps, num_comp_third, multilet_lines[2])
                        self.best_model_n_dict[multilet_lines[2]] = gaussian_1p_v(velocity_dict[line][2], self.x0_e, self.sigma_e, amps[2])
                        if broad_amp_3:
                            if (multilet_lines[2] in self.fit_func_choices.keys()) and (self.fit_func_choices[multilet_lines[2]] == 'Lorentzian') and (max_comp == 2):
                                self.best_model_b_dict[multilet_lines[2]] = lorentzian_1p_v(velocity_dict[line][2], self.x0_b, self.sigma_b, broad_amp_3[0])
                            else:
                                self.best_model_b_dict[multilet_lines[2]] = gaussian_1p_v(velocity_dict[line][2], self.x0_b, self.sigma_b, broad_amp_3[0])
                        params_line += broad_amp_1 + broad_amp_2 + broad_amp_3
                        if any(x == 3 for x in [num_comp_first, num_comp_second, num_comp_third]):
                            params_line += [self.x0_b2, self.sigma_b2] 
                            # line 1
                            if num_comp_first == 3:
                                broad_amp2_1 = self.get_broad_amp(self.best_amps, num_comp_first, multilet_lines[0], "2")
                                params_line += broad_amp2_1 
                                if broad_amp2_1:
                                    if (multilet_lines[0] in self.fit_func_choices.keys()) and (self.fit_func_choices[multilet_lines[0]] == 'Lorentzian'):
                                        self.best_model_b2_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], self.x0_b2, self.sigma_b2, broad_amp2_1[0])
                                    else:
                                        self.best_model_b2_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], self.x0_b2, self.sigma_b2, broad_amp2_1[0]) 
                            # line 2
                            if num_comp_second == 3:
                                broad_amp2_2 = self.get_broad_amp(self.best_amps, num_comp_second, multilet_lines[1], "2")
                                params_line += broad_amp2_2
                                if broad_amp2_2:
                                    if (multilet_lines[1] in self.fit_func_choices.keys()) and (self.fit_func_choices[multilet_lines[1]] == 'Lorentzian'):
                                        self.best_model_b2_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], self.x0_b2, self.sigma_b2, broad_amp2_2[0])
                                    else:
                                        self.best_model_b2_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], self.x0_b2, self.sigma_b2, broad_amp2_2[0])
                            # line 3
                            if num_comp_third == 3:
                                broad_amp2_3 = self.get_broad_amp(self.best_amps, num_comp_third, multilet_lines[2], "2")
                                params_line += broad_amp2_3
                                if broad_amp2_3:
                                    if (multilet_lines[2] in self.fit_func_choices.keys()) and (self.fit_func_choices[multilet_lines[2]] == 'Lorentzian'):
                                        self.best_model_b2_dict[multilet_lines[2]] = lorentzian_1p_v(velocity_dict[line][2], self.x0_b2, self.sigma_b2, broad_amp2_3[0])
                                    else:
                                        self.best_model_b2_dict[multilet_lines[2]] = gaussian_1p_v(velocity_dict[line][2], self.x0_b2, self.sigma_b2, broad_amp2_3[0])
                        # check whether any peak needs an absorption component
                        if any(absorption_check):
                            params_line += [self.x0_a, self.sigma_a] + [self.best_amps[l.split(' ')[1] + '_abs'] for l in multilet_lines if l in self.absorption_lines]
                        comp_first_abs, comp_second_abs, comp_third_abs = np.array([True] * 3) * absorption_check
                        if comp_first_abs:
                            abs_amp = [self.best_amps[f"{multilet_lines[0].split(' ')[1]}_abs"]]
                            if (multilet_lines[0] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[0]] == 'Lorentzian'):
                                self.best_model_ab_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model_ab_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], self.x0_a, self.sigma_a, abs_amp[0])
                        if comp_second_abs:
                            abs_amp = [self.best_amps[f"{multilet_lines[1].split(' ')[1]}_abs"]]
                            if (multilet_lines[1] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[1]] == 'Lorentzian'):
                                self.best_model_ab_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model_ab_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], self.x0_a, self.sigma_a, abs_amp[0])
                        if comp_third_abs:
                            abs_amp = [self.best_amps[f"{multilet_lines[2].split(' ')[1]}_abs"]]
                            if (multilet_lines[2] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[2]] == 'Lorentzian'):
                                self.best_model_ab_dict[multilet_lines[2]] = lorentzian_1p_v(velocity_dict[line][2], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model_ab_dict[multilet_lines[2]] = gaussian_1p_v(velocity_dict[line][2], self.x0_a, self.sigma_a, abs_amp[0])
                        abs_lorentz = True if ((multilet_lines[0] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[0]] == 'Lorentzian') or \
                                               (multilet_lines[1] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[1]] == 'Lorentzian') or \
                                               (multilet_lines[2] in self.fit_func_abs_choices.keys() and self.fit_func_abs_choices[multilet_lines[2]] == 'Lorentzian')) else False

                        # whether use the lorentzian function to fit broad wings or not
                        if ((multilet_lines[0] in self.fit_func_choices.keys()) and self.fit_func_choices[multilet_lines[0]] == 'Lorentzian') or \
                           ((multilet_lines[1] in self.fit_func_choices.keys()) and self.fit_func_choices[multilet_lines[1]] == 'Lorentzian') or \
                           ((multilet_lines[2] in self.fit_func_choices.keys()) and self.fit_func_choices[multilet_lines[2]] == 'Lorentzian'):
                            # best model
                            self.best_model[multilet_lines[0]], self.best_model[multilet_lines[1]], self.best_model[multilet_lines[2]], self.best_model[line] = \
                            gaussian_lorentzian_3p_v_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], velocity_dict[line][2],
                                                             num_comp_first, num_comp_second, num_comp_third, comp_first_abs, comp_second_abs, comp_third_abs, abs_lorentz)
                            # residual 
                            self.residual_dict[line] = residual_3p_gl_v_c_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                                                  velocity_dict[line][2], flux_v_dict[line], err_v_dict[line], 
                                                                                  num_comp_first, num_comp_second, num_comp_third, 
                                                                                  comp_first_abs, comp_second_abs, comp_third_abs, abs_lorentz)
                        else:
                            # best model
                            self.best_model[multilet_lines[0]], self.best_model[multilet_lines[1]], self.best_model[multilet_lines[2]], self.best_model[line] = \
                            gaussian_3p_v_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], velocity_dict[line][2],
                                                  num_comp_first, num_comp_second, num_comp_third, comp_first_abs, comp_second_abs, comp_third_abs, abs_lorentz)
                            # residual 
                            self.residual_dict[line] = residual_3p_v_c_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                                               velocity_dict[line][2], flux_v_dict[line], err_v_dict[line], 
                                                                               num_comp_first, num_comp_second, num_comp_third, 
                                                                               comp_first_abs, comp_second_abs, comp_third_abs, abs_lorentz)
            # for single line profile
            else:
                amp = [self.best_amps[line.split(' ')[1]]]
                # single emission component
                if (line not in self.absorption_lines) and (line not in self.broad_wings_lines):
                    params_line = [self.x0_e, self.sigma_e] + amp
                    # best model
                    self.best_model[line] = gaussian_1p_v(velocity_dict[line], self.x0_e, self.sigma_e, self.best_amps[line.split(' ')[1]])
                    # residual
                    self.residual_dict[line] = residual_1p_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                # multi emission components
                elif (line not in self.absorption_lines) and (line in self.broad_wings_lines):
                    # double emission components
                    if line in self.double_gauss_lines:
                        broad_amp = [self.best_amps[f"{line.split(' ')[1]}_b"]]
                        params_line = [self.x0_e, self.x0_b, self.sigma_e, self.sigma_b] + amp + broad_amp
                        if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian'):
                            # best model
                            self.best_model[line] = gaussian_lorentz_2p_v(velocity_dict[line], self.x0_e, self.x0_b, self.sigma_e, self.sigma_b, 
                                                                          self.best_amps[line.split(' ')[1]], self.best_amps[line.split(' ')[1]+'_b'])
                            self.best_model_n_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_e, self.sigma_e, amp[0])
                            self.best_model_b_dict[line] = lorentzian_1p_v(velocity_dict[line], self.x0_b, self.sigma_b, broad_amp[0])
                            # residual
                            self.residual_dict[line] = residual_2p_gl_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                        else:
                            # best model
                            self.best_model[line] = gaussian_2p_v(velocity_dict[line], self.x0_e, self.x0_b, self.sigma_e, self.sigma_b, 
                                                                  self.best_amps[line.split(' ')[1]], self.best_amps[line.split(' ')[1]+'_b'])
                            self.best_model_n_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_e, self.sigma_e, amp[0])
                            self.best_model_b_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_b, self.sigma_b, broad_amp[0])
                            # residual
                            self.residual_dict[line] = residual_2p_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                    # triple emission components
                    if line in self.triple_gauss_lines:
                        broad_amp = [self.best_amps[f"{line.split(' ')[1]}_b"], self.best_amps[f"{line.split(' ')[1]}_b2"]]
                        params_line = [self.x0_e, self.x0_b, self.x0_b2, self.sigma_e, self.sigma_b, self.sigma_b2] + amp + broad_amp
                        if (line in self.fit_func_choices.keys()) and (self.fit_func_choices[line] == 'Lorentzian'):
                            # best model
                            self.best_model[line] = gaussian_lorentz_3p_v(velocity_dict[line], self.x0_e, self.x0_b, self.x0_b2, self.sigma_e, self.sigma_b, self.sigma_b2,
                                                                          self.best_amps[line.split(' ')[1]], self.best_amps[line.split(' ')[1]+'_b'], self.best_amps[line.split(' ')[1]+'_b2'])
                            self.best_model_n_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_e, self.sigma_e, amp[0])
                            self.best_model_b_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_b, self.sigma_b, broad_amp[0])
                            self.best_model_b2_dict[line] = lorentzian_1p_v(velocity_dict[line], self.x0_b2, self.sigma_b2, broad_amp[1])
                            # residual
                            self.residual_dict[line] = residual_3p_gl_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                        else:
                            # best model
                            self.best_model[line] = gaussian_3p_v(velocity_dict[line], self.x0_e, self.x0_b, self.x0_b2, self.sigma_e, self.sigma_b, self.sigma_b2,
                                                                  self.best_amps[line.split(' ')[1]], self.best_amps[line.split(' ')[1]+'_b'], self.best_amps[line.split(' ')[1]+'_b2'])
                            self.best_model_n_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_e, self.sigma_e, amp[0])
                            self.best_model_b_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_b, self.sigma_b, broad_amp[0])
                            self.best_model_b2_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_b2, self.sigma_b2, broad_amp[1])
                            # residual
                            self.residual_dict[line] = residual_3p_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                # single line profile with emission+absorption components
                elif line in self.absorption_lines:
                    abs_amp = [self.best_amps[f"{line.split(' ')[1]}_abs"]]
                    if line not in self.broad_wings_lines:
                        params_line = [self.x0_e, self.x0_a, self.sigma_e, self.sigma_a] + amp + abs_amp
                        # self.best_model_n_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_e, self.sigma_e, amp[0])   
                        if (line in self.fit_func_abs_choices.keys()) and (self.fit_func_abs_choices[line] == 'Lorentzian'):
                            self.best_model[line] = gaussian_lorentz_2p_v(velocity_dict[line], self.x0_e, self.x0_a, self.sigma_e, self.sigma_a, 
                                                                          self.best_amps[line.split(' ')[1]], self.best_amps[line.split(' ')[1]+'_abs'])
                            self.residual_dict[line] = residual_2p_gl_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                            self.best_model_ab_dict[line] = lorentzian_1p_v(velocity_dict[line], self.x0_a, self.sigma_a, abs_amp[0])
                        else:
                            self.best_model[line] = gaussian_2p_v(velocity_dict[line], self.x0_e, self.x0_a, self.sigma_e, self.sigma_a, 
                                                                  self.best_amps[line.split(' ')[1]], self.best_amps[line.split(' ')[1]+'_abs'])
                            self.residual_dict[line] = residual_2p_gl_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                            self.best_model_ab_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_a, self.sigma_a, abs_amp[0])
                    else:
                        if line in self.double_gauss_lines:
                            broad_amp = [self.best_amps[f"{line.split(' ')[1]}_b"]]
                            params_line = [self.x0_e, self.x0_b, self.x0_a, self.sigma_e, self.sigma_b, self.sigma_a] + amp + broad_amp + abs_amp
                            self.best_model_n_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_e, self.sigma_e, amp[0])   
                            self.best_model_b_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_b, self.sigma_b, broad_amp[0]) 
                            if (line in self.fit_func_abs_choices.keys()) and (self.fit_func_abs_choices[line] == 'Lorentzian'):
                                self.best_model[line] = gaussian_lorentz_3p_v(velocity_dict[line], self.x0_e, self.x0_b, self.x0_a, self.sigma_e, self.sigma_b, self.sigma_a,
                                                                              self.best_amps[line.split(' ')[1]], self.best_amps[line.split(' ')[1]+'_b'], self.best_amps[line.split(' ')[1]+'_abs'])
                                self.residual_dict[line] = residual_3p_gl_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                                self.best_model_ab_dict[line] = lorentzian_1p_v(velocity_dict[line], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model[line] = gaussian_3p_v(velocity_dict[line], self.x0_e, self.x0_b, self.x0_a, self.sigma_e, self.sigma_b, self.sigma_a,
                                                                      self.best_amps[line.split(' ')[1]], self.best_amps[line.split(' ')[1]+'_b'], self.best_amps[line.split(' ')[1]+'_abs'])
                                self.residual_dict[line] = residual_3p_gl_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                                self.best_model_ab_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_a, self.sigma_a, abs_amp[0])
                        elif line in self.triple_gauss_lines:
                            broad_amp = [self.best_amps[f"{line.split(' ')[1]}_b"], self.best_amps[f"{line.split(' ')[1]}_b2"]]
                            params_line = [self.x0_e, self.x0_b, self.x0_b2, self.x0_a, self.sigma_e, self.sigma_b, self.sigma_b2, self.sigma_a] + amp + broad_amp + abs_amp
                            self.best_model_n_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_e, self.sigma_e, amp[0])   
                            self.best_model_b_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_b, self.sigma_b, broad_amp[0]) 
                            self.best_model_b2_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_b2, self.sigma_b2, broad_amp[1])
                            if (line in self.fit_func_abs_choices.keys()) and (self.fit_func_abs_choices[line] == 'Lorentzian'):
                                self.best_model[line] = gaussian_lorentz_4p_v(velocity_dict[line], self.x0_e, self.x0_b, self.x0_b2, self.x0_a, self.sigma_e, self.sigma_b, self.sigma_b2, self.sigma_a,
                                                                              self.best_amps[line.split(' ')[1]], self.best_amps[line.split(' ')[1]+'_b'], self.best_amps[line.split(' ')[1]+'_b2'],
                                                                              self.best_amps[line.split(' ')[1]+'_abs'])
                                self.residual_dict[line] = residual_4p_gl_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                                self.best_model_ab_dict[line] = lorentzian_1p_v(velocity_dict[line], self.x0_a, self.sigma_a, abs_amp[0])
                            else:
                                self.best_model[line] = gaussian_4p_v(velocity_dict[line], self.x0_e, self.x0_b, self.x0_b2, self.x0_a, self.sigma_e, self.sigma_b, self.sigma_b2, self.sigma_a,
                                                                      self.best_amps[line.split(' ')[1]], self.best_amps[line.split(' ')[1]+'_b'], self.best_amps[line.split(' ')[1]+'_b2'],
                                                                      self.best_amps[line.split(' ')[1]+'_abs'])
                                self.residual_dict[line] = residual_4p_gl_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                                self.best_model_ab_dict[line] = gaussian_1p_v(velocity_dict[line], self.x0_a, self.sigma_a, abs_amp[0])
            # number of parameters 
            self.params_num_dict[line] = len(params_line)
        ################# End: obtain the best-fitting model for each line #################
        return (self.best_model, self.residual_dict, self.best_chi2)


