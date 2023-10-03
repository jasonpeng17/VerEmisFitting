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

from line_fitting_exec import *
from line_fitting_prelim import *
from IPython import embed

class line_fitting_model():
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

    def extract_key_parts_from_ratio(self, ratio):
        """Helper function to extract the amplitude names from the dictionary that contains the fixed amplitude ratios"""
        parts = ratio.split('_over_')
        
        # Extract parts from the provided ratio string, preserving any suffixes.
        first_part = parts[0].split('_')[1:]
        second_part = parts[1].split('_')[0:]

        return ['_'.join(first_part), '_'.join(second_part)]

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
                if (line in broad_wings_lines):
                    # double emission components
                    if line in double_gauss_lines:
                        broad_amps = [params[f"amp_{line.split(' ')[1]}_b"]]
                        params_line = [x0_e, x0_b, sigma_e, sigma_b] + amps + broad_amps
                        residuals_all.append(residual_2p_v_c(params_line, x, y, yerr))
                    # triple emission components
                    if line in triple_gauss_lines:
                        broad_amps = [params[f"amp_{line.split(' ')[1]}_b"]] + [params[f"amp_{line.split(' ')[1]}_b2"]]
                        params_line = [x0_e, x0_b, x0_b2, sigma_e, sigma_b, sigma_b2] + amps + broad_amps
                        residuals_all.append(residual_3p_v_c(params_line, x, y, yerr))
                # emission + absorption components
                if (line in absorption_lines):
                    abs_amps = [params[f"amp_{line.split(' ')[1]}_abs"]]
                    params_line = [x0_e, x0_a, sigma_e, sigma_a] + amps + abs_amps
                    residuals_all.append(residual_2p_gl_v_c(params_line, x, y, yerr))
        return np.concatenate(residuals_all)


    def assign_best(self, model_dict, x0_e, sigma_e, amps, x0_b=None, sigma_b=None, x0_b2=None, sigma_b2=None, x0_a=None, sigma_a=None, 
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
        # best-fitting lmfit result
        self.best_res = self.result
        # best-fitting reduced chi2
        self.best_chi2 = self.result.redchi
        # best-fitting amps dictionary
        self.best_amps = self.amps
        # best-fitting model dict
        self.best_model = model_dict
        # best-fitting param dict
        self.best_param_dict = self.param_dict

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
        print(colored(f"Iteration #{self.current_iteration}: ", 'green', attrs=['bold', 'underline']))
        print(colored("The current best chi2 value is ", 'green'))
        print("{0:.5f}".format(self.best_chi2))
        print(colored("The current best parameter values are ", 'green'))
        print(self.best_params)

    def check_and_assign_best(self, model_dict, x0_e, sigma_e, amps, x0_b=None, sigma_b=None, x0_b2=None, sigma_b2=None,
                              x0_a=None, sigma_a=None, absorption=False, broad_wing=False, double_gauss=False, triple_gauss=False):
        """
        Check the model fitting conditions and assign the best parameters, result, and model if they meet the conditions.

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
            Whether the broad wing component exists
        absorption : bool
            Whether the absorption component exists
        double_gauss : bool
            Whether the broad-wing lines need a double gaussian model
        triple_gauss : bool
            Whether the broad-wing lines need a triple gaussian model
        """
        # conditions for each case
        conditions_no_absorption_no_broadwing = all((amps[line.split(' ')[1].split('&')[0]] > 0) and (amps[line.split(' ')[1].split('&')[1]] > 0) if '&' in line 
                                                     else amps[line.split(' ')[1]] > 0 for line in self.emission_lines) and (-200 < x0_e < 200)
        if broad_wing:
            if (not triple_gauss):
                conditions_broadwing = all((amps[line.split(' ')[1]] > amps[line.split(' ')[1]+'_b']) and (amps[line.split(' ')[1]+'_b'] > 0) \
                                       for line in self.broad_wings_lines) and (-200 < x0_e < 200) and (-200 < x0_b < 200)
                # if broad wing
                if self.double_gauss_broad:
                    conditions_broadwing = conditions_broadwing and (sigma_b > sigma_e)
            if triple_gauss:
                conditions_broadwing1 = all((amps[line.split(' ')[1]] > amps[line.split(' ')[1]+'_b']) and (amps[line.split(' ')[1]+'_b'] > 0) and \
                                           (amps[line.split(' ')[1]+'_b'] > amps[line.split(' ')[1]+'_b2']) and (amps[line.split(' ')[1]+'_b2'] > 0) \
                                           for line in self.triple_gauss_lines) 
                conditions_broadwing = conditions_broadwing1 and (-200 < x0_e < 200) and (-200 < x0_b < 200) and (-200 < x0_b2 < 200)
                # if broad wing
                if self.triple_gauss_broad:
                    conditions_broadwing = conditions_broadwing and (sigma_b > sigma_e) and (sigma_b2 > sigma_b)
                if double_gauss:
                    conditions_broadwing2 = all((amps[line.split(' ')[1]] > amps[line.split(' ')[1]+'_b']) and (amps[line.split(' ')[1]+'_b'] > 0) \
                                               for line in self.double_gauss_lines)
                    conditions_broadwing = conditions_broadwing1 and conditions_broadwing2 and (-200 < x0_e < 200) and (-200 < x0_b < 200) and (-200 < x0_b2 < 200) 
                    # if broad wing
                    if self.triple_gauss_broad:
                        conditions_broadwing = conditions_broadwing and (sigma_b > sigma_e) and (sigma_b2 > sigma_b)
        if absorption:
            conditions_absorption = all((amps[line.split(' ')[1]] > 0 and amps[line.split(' ')[1]+'_abs'] < 0) for line in self.absorption_lines) \
                                        and (sigma_a > sigma_e) and (-200 < x0_e < 200) and (-200 < x0_a < 200)
            # conditions_absorption = all((amps[line.split(' ')[1]] > 0 and amps[line.split(' ')[1]+'_abs'] < 0) for line in self.absorption_lines)
        if broad_wing and absorption:
            conditions_absorption_broadwing = conditions_broadwing and conditions_absorption

        if (not absorption) and (not broad_wing) and conditions_no_absorption_no_broadwing:
            self.assign_best(model_dict, x0_e, sigma_e, amps)
        elif (not absorption) and broad_wing and conditions_broadwing:
            self.assign_best(model_dict, x0_e, sigma_e, amps, x0_b=x0_b, sigma_b=sigma_b, x0_b2=x0_b2, sigma_b2=sigma_b2, broad_wing=True, 
                             double_gauss=double_gauss, triple_gauss=triple_gauss)
        elif absorption and (not broad_wing) and conditions_absorption:
            self.assign_best(model_dict, x0_e, sigma_e, amps, x0_a=x0_a, sigma_a=sigma_a, absorption=True)
        elif absorption and broad_wing and conditions_absorption_broadwing:
            self.assign_best(model_dict, x0_e, sigma_e, amps, x0_a=x0_a, sigma_a=sigma_a, x0_b=x0_b, sigma_b=sigma_b, x0_b2=x0_b2, sigma_b2=sigma_b2,
                             absorption=True, broad_wing=True, double_gauss=double_gauss, triple_gauss=triple_gauss)


    def fitting_all_lines(self, input_arr, n_iteration = 1000):
        """
        Fit all intended emission lines in the velocity space using a specified number of iterations.

        Parameters
        ----------
        input_arr : list
            A list containing velocity_arr, flux_v_arr, err_v_arr, initial_guess, and param_range.
        n_iteration : int, optional
            Number of iterations for fitting, default is 1000.
        Returns
        -------
        best_model (array): The best fitting model found after iterating through the fitting process.
        best_chi2 (float): The reduced chi-squared value corresponding to the best fitting model.
        """
        # fit all intended emission lines in the velocity space
        velocity_dict, flux_v_dict, err_v_dict, initial_guess_dict, param_range_dict, amps_ratio_dict, self.absorption_lines, self.broad_wings_lines, \
        self.double_gauss_lines, self.triple_gauss_lines, self.double_gauss_broad, self.triple_gauss_broad, fitting_method = input_arr

        # get a copy of the initial guess dict
        initial_guess_dict_old = initial_guess_dict.copy()

        # all intended emission lines
        self.emission_lines = list(velocity_dict.keys())

        # define the lists that contains the fixed amplitude pairs and their fixed amp ratios
        self.amps_fixed_list = [part for ratio, value in amps_ratio_dict.items() for part in self.extract_key_parts_from_ratio(ratio)]
        self.amps_ratio_list = [value for ratio, value in amps_ratio_dict.items()]

        # determine whether to fix the velocity center and width for multi-component fittings
        fitting_methods = {'Free fitting': (True, True), 'Fix velocity centroid': (False, True), 'Fix velocity width': (True, False), 
                           'Fix velocity centroid and width': (False, False)}
        vary_center, vary_width = fitting_methods[fitting_method]
        vary_dict = {'e': (True, True), 'b': (vary_center, vary_width), 'b2': (vary_center, vary_width),  'a': (vary_center, vary_width)}

        for i in range(n_iteration):
            # define the current iteration number
            self.current_iteration = i + 1
            # define the input parameters
            self.params = Parameters()
            # velocity center and width of the narrow emission component
            self.params.add('center_e', value=initial_guess_dict['v_e'][0], vary = True)
            self.params.add('sigma_e', value=initial_guess_dict['v_e'][1], min = 0, max = 200, vary = True)
            for component in ['b', 'b2', 'a']:
                # define a max sigma value 
                sigma_max = 900 if component == 'a' else 900
                # set initial values only for other velocity info
                if f'v_{component}' in initial_guess_dict:
                    self.params.add(f"center_{component}", value=initial_guess_dict[f'v_{component}'][0],
                                    expr=f"center_e" if not vary_dict[component][0] else None)
                    self.params.add(f"sigma_{component}", value=initial_guess_dict[f'v_{component}'][1], min=0, max=sigma_max, 
                                    expr=f"sigma_e" if not vary_dict[component][1] else None)

            # set initial values only for amplitude info (free amplitude fitting)
            for line, initial_guess in initial_guess_dict.items():
                ion_wave_split = line.split(' ')
                if len(ion_wave_split) > 1: 
                    # lines that follow fixed amp ratio fitting strategy
                    if ion_wave_split[1] in self.amps_fixed_list:
                        indx_num = self.amps_fixed_list.index(ion_wave_split[1])
                        if indx_num % 2 != 0:
                            amp_ratio_indx = int((indx_num + 1) / 2 - 1)
                            amp_ratio = self.amps_ratio_list[indx_num - 1] # fixed amp ratio between these two lines
                            self.params.add(f"amp_{ion_wave_split[1]}", value=initial_guess[0]) # first line 
                            self.params.add(f"amp_{self.amps_fixed_list[indx_num - 1]}", expr = f"{amp_ratio} * amp_{ion_wave_split[1]}") # second line

                        if (line in self.broad_wings_lines):
                            # check whether the lines' second velocity component has fixed ratio or not
                            if f"{ion_wave_split[1]}_b" in self.amps_fixed_list:
                                indx_num_b = self.amps_fixed_list.index(f"{ion_wave_split[1]}_b")
                                if indx_num_b % 2 != 0:
                                    amp_ratio_b_indx = int((indx_num_b + 1) / 2 - 1)
                                    amp_ratio_b = self.amps_ratio_list[amp_ratio_b_indx] # fixed amp ratio between these two lines
                                    self.params.add(f"amp_{ion_wave_split[1]}_b", value=initial_guess[1]) # first line 
                                    self.params.add(f"amp_{self.amps_fixed_list[indx_num_b - 1]}", expr=f"{amp_ratio_b} * amp_{ion_wave_split[1]}_b") # second line
                            else:
                                self.params.add(f"amp_{ion_wave_split[1]}_b", value=initial_guess[1])
                                
                        if (line in self.triple_gauss_lines):
                            # check whether the lines' third velocity component has fixed ratio or not
                            if f"{ion_wave_split[1]}_b2" in self.amps_fixed_list:
                                indx_num_b2 = self.amps_fixed_list.index(f"{ion_wave_split[1]}_b2")
                                if indx_num_b2 % 2 != 0:
                                    amp_ratio_b2_indx = int((indx_num_b2 + 1) / 2 - 1)
                                    amp_ratio_b2 = self.amps_ratio_list[amp_ratio_b2_indx] # fixed amp ratio between these two lines
                                    self.params.add(f"amp_{ion_wave_split[1]}_b2", value=initial_guess[2]) # first line 
                                    self.params.add(f"amp_{self.amps_fixed_list[indx_num_b2 - 1]}", expr = f"{amp_ratio_b2} * amp_{ion_wave_split[1]}_b2") # second line
                            else:
                                self.params.add(f"amp_{ion_wave_split[1]}_b2", value=initial_guess[2])

                    # lines that follow free amplitude fitting strategy
                    if ion_wave_split[1] not in self.amps_fixed_list:
                        self.params.add(f"amp_{ion_wave_split[1]}", value=initial_guess[0])
                        if line in self.broad_wings_lines:
                            self.params.add(f"amp_{ion_wave_split[1]}_b", value=initial_guess[1])
                            if line in self.triple_gauss_lines:
                                self.params.add(f"amp_{ion_wave_split[1]}_b2", value=initial_guess[2])
                        if line in self.absorption_lines:
                            self.params.add(f"amp_{ion_wave_split[1]}_abs", value=initial_guess[1])

            # obtain the best result of this iteration
            self.result = minimize(self.residual_v_f_all, self.params, args=(velocity_dict, flux_v_dict, err_v_dict, self.absorption_lines, self.broad_wings_lines, 
                                                                             self.double_gauss_lines, self.triple_gauss_lines))
            self.param_dict = self.result.params.valuesdict()
            # obtain the best-fitting velocity center and width of each component
            x0_e = self.param_dict["center_e"]
            sigma_e = self.param_dict["sigma_e"]
            x0_a = self.param_dict.get("center_a", None)
            sigma_a = self.param_dict.get("sigma_a", None)
            x0_b = self.param_dict.get("center_b", None)
            sigma_b = self.param_dict.get("sigma_b", None)
            x0_b2 = self.param_dict.get("center_b2", None)
            sigma_b2 = self.param_dict.get("sigma_b2", None)

            # collect amplitude values
            self.amps = {key.replace('amp_', ''): value for key, value in self.param_dict.items() if 'amp' in key}
               
            # model for each component
            model_dict = {}
            for line in velocity_dict.keys():
                # doublet that needs to be fitted together
                if '&' in line:
                    multilet_lines = split_multilet_line(line)
                    amps = [self.amps[l.split(' ')[1]] for l in multilet_lines]
                    # single emission component
                    if all((l not in self.broad_wings_lines) for l in multilet_lines):
                        params_line = [x0_e, sigma_e] + amps
                        # doublet
                        if len(multilet_lines) == 2:
                            model_dict[multilet_lines[0]], model_dict[multilet_lines[1]], model_dict[line] = gaussian_2p_v_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1])
                        # triplet
                        if len(multilet_lines) == 3:
                            model_dict[multilet_lines[0]], model_dict[multilet_lines[1]], model_dict[multilet_lines[2]], model_dict[line] = gaussian_3p_v_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], velocity_dict[line][2])
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
                        params_line = [x0_e, sigma_e] + amps + [x0_b, sigma_b]

                        # Double line profiles
                        if len(multilet_lines) == 2:
                            broad_amp_1 = self.get_broad_amp(self.amps, num_comp_first, multilet_lines[0])
                            broad_amp_2 = self.get_broad_amp(self.amps, num_comp_second, multilet_lines[1])
                            params_line += broad_amp_1 + broad_amp_2
                            if num_comp_first == 3 or num_comp_second == 3:
                                params_line += [x0_b2, sigma_b2] 
                                if num_comp_first == 3:
                                    params_line += self.get_broad_amp(self.amps, num_comp_first, multilet_lines[0], "2") 
                                if num_comp_second == 3:
                                    params_line += self.get_broad_amp(self.amps, num_comp_second, multilet_lines[1], "2")
                            # append the line model to the model dict
                            model_dict[multilet_lines[0]], model_dict[multilet_lines[1]], model_dict[line] = \
                            gaussian_2p_v_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                  num_comp_first=num_comp_first, num_comp_second=num_comp_second)

                        # Triple line profiles
                        if len(multilet_lines) == 3:
                            broad_amp_1 = self.get_broad_amp(self.amps, num_comp_first, multilet_lines[0])
                            broad_amp_2 = self.get_broad_amp(self.amps, num_comp_second, multilet_lines[1])
                            broad_amp_3 = self.get_broad_amp(self.amps, num_comp_third, multilet_lines[2])
                            params_line += broad_amp_1 + broad_amp_2 + broad_amp_3
                            if any(x == 3 for x in [num_comp_first, num_comp_second, num_comp_third]):
                                params_line += [x0_b2, sigma_b2] 
                                if num_comp_first == 3:
                                    params_line += self.get_broad_amp(self.amps, num_comp_first, multilet_lines[0], "2") 
                                if num_comp_second == 3:
                                    params_line += self.get_broad_amp(self.amps, num_comp_second, multilet_lines[1], "2")
                                if num_comp_third == 3:
                                    params_line += self.get_broad_amp(self.amps, num_comp_third, multilet_lines[2], "2")
                            # append the line model to the model dict
                            model_dict[multilet_lines[0]], model_dict[multilet_lines[1]], model_dict[multilet_lines[2]], model_dict[line] = \
                            gaussian_3p_v_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], velocity_dict[line][2],
                                                  num_comp_first=num_comp_first, num_comp_second=num_comp_second, num_comp_third=num_comp_third)

                # for single line profile
                else:
                    # single emission component
                    if (line not in self.absorption_lines) and (line not in self.broad_wings_lines):
                        model_dict[line] = gaussian_1p_v(velocity_dict[line], x0_e, sigma_e, self.amps[line.split(' ')[1]])
                    # multi emission components
                    elif (line not in self.absorption_lines) and (line in self.broad_wings_lines):
                        # double emission components
                        if line in self.double_gauss_lines:
                            model_dict[line] = gaussian_2p_v(velocity_dict[line], x0_e, x0_b, sigma_e, sigma_b, 
                                                             self.amps[line.split(' ')[1]], self.amps[line.split(' ')[1]+'_b'])
                        # triple emission components
                        if line in self.triple_gauss_lines:
                            model_dict[line] = gaussian_3p_v(velocity_dict[line], x0_e, x0_b, x0_b2, sigma_e, sigma_b, sigma_b2,
                                                             self.amps[line.split(' ')[1]], self.amps[line.split(' ')[1]+'_b'], self.amps[line.split(' ')[1]+'_b2'])
                    # emission + absorption components
                    elif (line in self.absorption_lines) and (line not in self.broad_wings_lines):
                        model_dict[line] = gaussian_lorentz_2p_v(velocity_dict[line], x0_e, x0_a, sigma_e, sigma_a, 
                                                                 self.amps[line.split(' ')[1]], self.amps[line.split(' ')[1]+'_abs'])
            # print(self.result.redchi)
            try:
                if (self.best_chi2 > self.result.redchi):
                    # print and assign the best-fitting parameters (and also print the best chi2 value)
                    self.check_and_assign_best(model_dict, x0_e, sigma_e, self.amps, x0_b=x0_b, sigma_b=sigma_b, x0_a=x0_a, sigma_a=sigma_a, x0_b2 = x0_b2, sigma_b2 = sigma_b2,
                                               absorption=bool(self.absorption_lines), broad_wing=bool(self.broad_wings_lines),
                                               double_gauss=bool(self.double_gauss_lines), triple_gauss=bool(self.triple_gauss_lines))
            except (UnboundLocalError, NameError, AttributeError) as e:
                # print and assign the best-fitting parameters (and also print the best chi2 value)
                self.check_and_assign_best(model_dict, x0_e, sigma_e, self.amps, x0_b=x0_b, sigma_b=sigma_b, x0_a=x0_a, sigma_a=sigma_a, x0_b2 = x0_b2, sigma_b2 = sigma_b2,
                                           absorption=bool(self.absorption_lines), broad_wing=bool(self.broad_wings_lines),
                                           double_gauss=bool(self.double_gauss_lines), triple_gauss=bool(self.triple_gauss_lines))
            # update the initial guess dict based on the parameter range dict
            initial_guess_dict = {key: value + np.float64((2*self.rng.random(1)-1))*param_range_dict[key] for key, value in initial_guess_dict_old.items()}

        # print(n_iteration)
        return (self.best_model, self.best_chi2)


