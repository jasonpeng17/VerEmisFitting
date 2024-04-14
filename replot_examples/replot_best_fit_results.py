# plot the best-fitting result based on the returned best-fit parameter and continuum table
import numpy as np
from astropy.io import fits
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from IPython import embed
from matplotlib import rc

# Add the parent directory to sys.path
parent_dir = os.path.abspath('../')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fitting_window_gui import *
from analysis_utils import *
from data_loading_utils import *
from modeling_utils import *
from line_fitting_model import *

c = const.c.to('km/s').value # speed of light

def extract_fit_window(wave, spec, espec, selected_line, redshift, folder_name, fit_cont_order = 1, vac_or_air = 'air'):
    '''
    Extract and return velocity array, flux array (in velocity space), and error array (in velocity space)
    for the given selected line or multiplet.
    '''
    if vac_or_air == 'air':
        elements_dict = read_wavelengths_from_file('../doc/air_wavelengths.txt', redshift)
    if vac_or_air == 'vac':
        elements_dict = read_wavelengths_from_file('../doc/vac_wavelengths.txt', redshift)
    wave_dict = merge_element_waves(elements_dict)
    line_waves = wave_dict[selected_line]

    # Calculate the central wavelength for velocity transformation
    # For a multiplet, use the average of the wavelengths
    central_wave = np.mean(line_waves)

    nan_index = np.isnan(spec) | np.isnan(espec)  # nan-value index
    # copy of wave, spec, and espec
    wave_c = np.copy(wave[np.logical_not(nan_index)])
    spec_c = np.copy(spec[np.logical_not(nan_index)])
    espec_c = np.copy(espec[np.logical_not(nan_index)])

    # determine the local continuum and subtract it from the flux array
    Fitting_Window = FittingWindow(wave_c, spec_c, folder_name = folder_name, line_name = selected_line, indx = 1) # a random indx number (!= 0) to not start bokeh server
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
    cont_f_fit, cont_f_fit_err, linecoeff, covar = region_around_line(wave_fit, flux_fit, cont_fit, order = fit_cont_order)[-4:]
    cont_return = np.array([cont_f_fit, cont_f_fit_err])
    # local-continuum-subtracted spectrum
    flux_sc_fit = flux_fit - cont_f_fit
    # error propagation
    err_sc_fit = np.sqrt((err_fit)**2 + (cont_f_fit_err)**2)

    # Choose the appropriate line wave based on the length of line_waves
    chosen_line_wave = line_waves[0] if len(line_waves) <= 2 else line_waves[1]

    # Transform wavelength array to velocity array for each line
    v_fit = [(wave_fit / w - 1) * c for w in line_waves]
    # Transform continuum-subtracted flux array to velocity space
    flux_sc_v_fit = [flux_sc_fit * chosen_line_wave / c]
    # Transform continuum-subtracted error array to velocity space
    err_sc_v_fit = [err_sc_fit * chosen_line_wave / c]
    
    # check whether there are masked regions
    if len(lmasks) >= 1: # if yes, then append the unmasked velocity and flux (no continuum subtracted) arrays 
        v_fit_c = [(wave_fit_c / w - 1) * c for w in line_waves]
        flux_v_fit_c = [flux_fit_c * chosen_line_wave / c]
        err_v_fit_c = [err_fit_c * chosen_line_wave / c]
        result_fit = v_fit + flux_sc_v_fit + err_sc_v_fit + v_fit_c + flux_v_fit_c + err_v_fit_c
    else:
        # if no, then append the masked velocity and flux (no continuum subtracted) arrays again
        flux_v_fit = [flux_fit * chosen_line_wave / c]
        err_v_fit = [err_fit * chosen_line_wave / c]
        result_fit = v_fit + flux_sc_v_fit + err_sc_v_fit + v_fit + flux_v_fit + err_v_fit

    return result_fit, cont_return, lmasks

def return_best_models(selected_line, fitting_method, broad_wings_lines, double_gauss_lines, triple_gauss_lines, 
                       fit_func_choices, absorption_lines, x0_dict, sigma_dict, amps_dict, amps_fixed_names):
    # initialize the best-fitting model and residual dicts
    model_dict = dict()
    residual_dict = dict()

    # reduced chi-square dictionary
    redchi2_dict = dict() 

    # model for narrow and broad emission components
    # if broad_wings_lines:
    best_model_n_dict = dict()
    best_model_b_dict = dict()
    # if triple_gauss_lines:
    best_model_b2_dict = dict()
    # model for emission and absorption components
    # if absorption_lines:
    best_model_em_dict = dict()
    best_model_ab_dict = dict()

    # unpack x0_dict and sigma_dict 
    x0_e, x0_b, x0_b2, x0_a = x0_dict['x0_e'], x0_dict['x0_b'], x0_dict['x0_b2'], x0_dict['x0_a']
    sigma_e, sigma_b, sigma_b2, sigma_a = sigma_dict['sigma_e'], sigma_dict['sigma_b'], sigma_dict['sigma_b2'], sigma_dict['sigma_a']

    # initialize the line_fitting_model class object
    line_fit_model = line_fitting_model()

    # number of fixed velocity parameters for each self.fitting_method (times num of velocity components for each line profile)
    vel_fixed_num = {'Free fitting': 0, 'Fix velocity centroid': 1, 'Fix velocity width': 1, 'Fix velocity centroid and width': 2}

    # iterate through each line and return their residuals and best-fitting models
    for line in selected_lines:
        if '&' in line:  # Special case for doublet that should be fitted together
            multilet_lines = split_multilet_line(line)
            amps = [amps_dict[key] for key in line.split(' ')[1].split('&')]
            # single emission component
            if all((l not in broad_wings_lines) for l in multilet_lines):
                params_line = [x0_e, sigma_e] + amps
                # doublet
                if len(multilet_lines) == 2:
                    model_dict[multilet_lines[0]], model_dict[multilet_lines[1]], model_dict[line] = gaussian_2p_v_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1])
                    residual_dict[line] = residual_2p_v_c_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                                  flux_v_dict[line], err_v_dict[line])
                # triplet
                elif len(multilet_lines) == 3:
                    model_dict[multilet_lines[0]], model_dict[multilet_lines[1]], model_dict[multilet_lines[2]], model_dict[line] = gaussian_3p_v_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], velocity_dict[line][2])
                    residual_dict[line] = residual_3p_v_c_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], velocity_dict[line][2],
                                                                  flux_v_dict[line], err_v_dict[line])
            # multi emission component
            else:
                for num_ii, l in enumerate(multilet_lines):
                    # Default single emission component
                    value = 1
                    # Check for double and triple emission components
                    if l in broad_wings_lines:
                        value = 2 if l not in triple_gauss_lines else 3
                    # Assign the determined value based on num_ii
                    if num_ii == 0:
                        num_comp_first = value
                    elif num_ii == 1:
                        num_comp_second = value
                    elif num_ii == 2:
                        num_comp_third = value
                
                # define the base of params 
                params_line = [x0_e, sigma_e] + amps + [x0_b, sigma_b]

                # Double line profiles
                if len(multilet_lines) == 2:
                    max_comp = np.max([num_comp_first, num_comp_second]) # maximum velocity component of two peaks 
                    # line 1
                    broad_amp_1 = line_fit_model.get_broad_amp(amps_dict, num_comp_first, multilet_lines[0])
                    best_model_n_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], x0_e, sigma_e, amps[0])
                    if broad_amp_1:
                        if (multilet_lines[0] in fit_func_choices.keys()) and (fit_func_choices[multilet_lines[0]] == 'Lorentzian') and (max_comp == 2):
                            best_model_b_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], x0_b, sigma_b, broad_amp_1[0])
                        else:
                            best_model_b_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], x0_b, sigma_b, broad_amp_1[0])
                    # line 2
                    broad_amp_2 = line_fit_model.get_broad_amp(amps_dict, num_comp_second, multilet_lines[1])
                    best_model_n_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], x0_e, sigma_e, amps[1])
                    if broad_amp_2:
                        if (multilet_lines[1] in fit_func_choices.keys()) and (fit_func_choices[multilet_lines[1]] == 'Lorentzian') and (max_comp == 2):
                            best_model_b_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], x0_b, sigma_b, broad_amp_2[0])
                        else:
                            best_model_b_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], x0_b, sigma_b, broad_amp_2[0])
                    params_line += broad_amp_1 + broad_amp_2
                    # check whether they have the third emission components
                    if num_comp_first == 3 or num_comp_second == 3:
                        params_line += [x0_b2, sigma_b2] 
                        # line 1
                        if num_comp_first == 3:
                            broad_amp2_1 = line_fit_model.get_broad_amp(amps_dict, num_comp_first, multilet_lines[0], "2")
                            params_line += broad_amp2_1
                            if broad_amp2_1:
                                if (multilet_lines[0] in fit_func_choices.keys()) and (fit_func_choices[multilet_lines[0]] == 'Lorentzian'):
                                    best_model_b2_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], x0_b2, sigma_b2, broad_amp2_1[0])
                                else:
                                    best_model_b2_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], x0_b2, sigma_b2, broad_amp2_1[0])
                        # line 2
                        if num_comp_second == 3:
                            broad_amp2_2 = line_fit_model.get_broad_amp(amps_dict, num_comp_second, multilet_lines[1], "2")
                            params_line += broad_amp2_2
                            if broad_amp2_2:
                                if (multilet_lines[1] in fit_func_choices.keys()) and (fit_func_choices[multilet_lines[1]] == 'Lorentzian'):
                                    best_model_b2_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], x0_b2, sigma_b2, broad_amp2_2[0])
                                else:
                                    best_model_b2_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], x0_b2, sigma_b2, broad_amp2_2[0])
                    # append the line residual to the residual dict
                    if ((multilet_lines[0] in fit_func_choices.keys()) and fit_func_choices[multilet_lines[0]] == 'Lorentzian') or \
                       ((multilet_lines[1] in fit_func_choices.keys()) and fit_func_choices[multilet_lines[1]] == 'Lorentzian'):
                        residual_dict[line] = residual_2p_gl_v_c_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                                              flux_v_dict[line], err_v_dict[line], 
                                                                              num_comp_first=num_comp_first, num_comp_second=num_comp_second)
                        model_dict[multilet_lines[0]], model_dict[multilet_lines[1]], model_dict[line] = \
                        gaussian_lorentzian_2p_v_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                         num_comp_first=num_comp_first, num_comp_second=num_comp_second)
                    else:
                        residual_dict[line] = residual_2p_v_c_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                                           flux_v_dict[line], err_v_dict[line], 
                                                                           num_comp_first=num_comp_first, num_comp_second=num_comp_second)
                        model_dict[multilet_lines[0]], model_dict[multilet_lines[1]], model_dict[line] = \
                        gaussian_2p_v_doublet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                              num_comp_first=num_comp_first, num_comp_second=num_comp_second)

                # Triple line profiles
                if len(multilet_lines) == 3:
                    max_comp = np.max([num_comp_first, num_comp_second, num_comp_third]) # maximum velocity component of three peaks 
                    # line 1
                    broad_amp_1 = line_fit_model.get_broad_amp(amps_dict, num_comp_first, multilet_lines[0])
                    best_model_n_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], x0_e, sigma_e, amps[0])
                    if broad_amp_1:
                        if (multilet_lines[0] in fit_func_choices.keys()) and (fit_func_choices[multilet_lines[0]] == 'Lorentzian') and (max_comp == 2):
                            best_model_b_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], x0_b, sigma_b, broad_amp_1[0])
                        else:
                            best_model_b_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], x0_b, sigma_b, broad_amp_1[0])
                    # line 2
                    broad_amp_2 = line_fit_model.get_broad_amp(amps_dict, num_comp_second, multilet_lines[1])
                    best_model_n_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], x0_e, sigma_e, amps[1])
                    if broad_amp_2:
                        if (multilet_lines[1] in fit_func_choices.keys()) and (fit_func_choices[multilet_lines[1]] == 'Lorentzian') and (max_comp == 2):
                            best_model_b_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], x0_b, sigma_b, broad_amp_2[0])
                        else:
                            best_model_b_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], x0_b, sigma_b, broad_amp_2[0])
                    # line 3
                    broad_amp_3 = line_fit_model.get_broad_amp(amps_dict, num_comp_third, multilet_lines[2])
                    best_model_n_dict[multilet_lines[2]] = gaussian_1p_v(velocity_dict[line][2], x0_e, sigma_e, amps[2])
                    if broad_amp_3:
                        if (multilet_lines[2] in fit_func_choices.keys()) and (fit_func_choices[multilet_lines[2]] == 'Lorentzian') and (max_comp == 2):
                            best_model_b_dict[multilet_lines[2]] = lorentzian_1p_v(velocity_dict[line][2], x0_b, sigma_b, broad_amp_3[0])
                        else:
                            best_model_b_dict[multilet_lines[2]] = gaussian_1p_v(velocity_dict[line][2], x0_b, sigma_b, broad_amp_3[0])
                    params_line += broad_amp_1 + broad_amp_2 + broad_amp_3
                    # check whether they have the third emission components
                    if any(x == 3 for x in [num_comp_first, num_comp_second, num_comp_third]):
                        params_line += [x0_b2, sigma_b2] 
                        # line 1
                        if num_comp_first == 3:
                            broad_amp2_1 = line_fit_model.get_broad_amp(amps_dict, num_comp_first, multilet_lines[0], "2")
                            params_line += broad_amp2_1 
                            if broad_amp2_1:
                                if (multilet_lines[0] in fit_func_choices.keys()) and (fit_func_choices[multilet_lines[0]] == 'Lorentzian'):
                                    best_model_b2_dict[multilet_lines[0]] = lorentzian_1p_v(velocity_dict[line][0], x0_b2, sigma_b2, broad_amp2_1[0])
                                else:
                                    best_model_b2_dict[multilet_lines[0]] = gaussian_1p_v(velocity_dict[line][0], x0_b2, sigma_b2, broad_amp2_1[0])
                        # line 2
                        if num_comp_second == 3:
                            broad_amp2_2 = line_fit_model.get_broad_amp(amps_dict, num_comp_second, multilet_lines[1], "2")
                            params_line += broad_amp2_2
                            if broad_amp2_2:
                                if (multilet_lines[1] in fit_func_choices.keys()) and (fit_func_choices[multilet_lines[1]] == 'Lorentzian'):
                                    best_model_b2_dict[multilet_lines[1]] = lorentzian_1p_v(velocity_dict[line][1], x0_b2, sigma_b2, broad_amp2_2[0])
                                else:
                                    best_model_b2_dict[multilet_lines[1]] = gaussian_1p_v(velocity_dict[line][1], x0_b2, sigma_b2, broad_amp2_2[0])
                        # line 3
                        if num_comp_third == 3:
                            broad_amp2_3 = line_fit_model.get_broad_amp(amps_dict, num_comp_third, multilet_lines[2], "2")
                            params_line += broad_amp2_3
                            if broad_amp2_3:
                                if (multilet_lines[2] in fit_func_choices.keys()) and (fit_func_choices[multilet_lines[2]] == 'Lorentzian'):
                                    best_model_b2_dict[multilet_lines[2]] = lorentzian_1p_v(velocity_dict[line][2], x0_b2, sigma_b2, broad_amp2_3[0])
                                else:
                                    best_model_b2_dict[multilet_lines[2]] = gaussian_1p_v(velocity_dict[line][2], x0_b2, sigma_b2, broad_amp2_3[0])
                    # append the line residual to the residual dict
                    if ((multilet_lines[0] in fit_func_choices.keys()) and fit_func_choices[multilet_lines[0]] == 'Lorentzian') or \
                       ((multilet_lines[1] in fit_func_choices.keys()) and fit_func_choices[multilet_lines[1]] == 'Lorentzian') or \
                       ((multilet_lines[2] in fit_func_choices.keys()) and fit_func_choices[multilet_lines[2]] == 'Lorentzian'):
                        residual_dict[line] = residual_3p_gl_v_c_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                                         velocity_dict[line][2], flux_v_dict[line], err_v_dict[line], 
                                                                         num_comp_first=num_comp_first, num_comp_second=num_comp_second, num_comp_third=num_comp_third)
                        model_dict[multilet_lines[0]], model_dict[multilet_lines[1]], model_dict[multilet_lines[2]], model_dict[line] = \
                        gaussian_lorentzian_3p_v_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], velocity_dict[line][2],
                                                         num_comp_first=num_comp_first, num_comp_second=num_comp_second, num_comp_third=num_comp_third)
                    else:
                        residual_dict[line] = residual_3p_v_c_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], 
                                                                      velocity_dict[line][2], flux_v_dict[line], err_v_dict[line], 
                                                                      num_comp_first=num_comp_first, num_comp_second=num_comp_second, num_comp_third=num_comp_third)
                        model_dict[multilet_lines[0]], model_dict[multilet_lines[1]], model_dict[multilet_lines[2]], model_dict[line] = \
                        gaussian_3p_v_triplet(params_line, velocity_dict[line][0], velocity_dict[line][1], velocity_dict[line][2],
                                              num_comp_first=num_comp_first, num_comp_second=num_comp_second, num_comp_third=num_comp_third)
        else:  # General case
            amp = [amps_dict[line.split(' ')[1]]]
            # single line profile with single emission component
            if (line not in absorption_lines) and (line not in broad_wings_lines):
                params_line = [x0_e, sigma_e] + amp
                residual_dict[line] = residual_1p_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                model_dict[line] = gaussian_1p_v(velocity_dict[line], x0_e, sigma_e, amp[0])
            # single line profile with multi emission components
            if (line not in absorption_lines) and (line in broad_wings_lines):
                # double emission components
                if line in double_gauss_lines:
                    broad_amp = [amps_dict[f"{line.split(' ')[1]}_b"]]
                    params_line = [x0_e, x0_b, sigma_e, sigma_b] + amp + broad_amp
                    if (line in fit_func_choices.keys()) and (fit_func_choices[line] == 'Lorentzian'):
                        residual_dict[line] = residual_2p_gl_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                        best_model_n_dict[line] = gaussian_1p_v(velocity_dict[line], x0_e, sigma_e, amp[0])
                        best_model_b_dict[line] = lorentzian_1p_v(velocity_dict[line], x0_b, sigma_b, broad_amp[0])
                        model_dict[line] = gaussian_lorentz_2p_v(velocity_dict[line], x0_e, x0_b, sigma_e, sigma_b, 
                                                                 amp[0], broad_amp[0])
                    else:
                        residual_dict[line] = residual_2p_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                        best_model_n_dict[line] = gaussian_1p_v(velocity_dict[line], x0_e, sigma_e, amp[0])
                        best_model_b_dict[line] = gaussian_1p_v(velocity_dict[line], x0_b, sigma_b, broad_amp[0])
                        model_dict[line] = gaussian_2p_v(velocity_dict[line], x0_e, x0_b, sigma_e, sigma_b, 
                                                         amp[0], broad_amp[0])
                # triple emission components
                if line in triple_gauss_lines:
                    broad_amp = [amps_dict[f"{line.split(' ')[1]}_b"], amps_dict[f"{line.split(' ')[1]}_b2"]]
                    params_line = [x0_e, x0_b, x0_b2, sigma_e, sigma_b, sigma_b2] + amp + broad_amp
                    if (line in fit_func_choices.keys()) and (fit_func_choices[line] == 'Lorentzian'):
                        residual_dict[line] = residual_3p_gl_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                        best_model_n_dict[line] = gaussian_1p_v(velocity_dict[line], x0_e, sigma_e, amp[0])
                        best_model_b_dict[line] = gaussian_1p_v(velocity_dict[line], x0_b, sigma_b, broad_amp[0])
                        best_model_b2_dict[line] = lorentzian_1p_v(velocity_dict[line], x0_b2, sigma_b2, broad_amp[1])
                        model_dict[line] = gaussian_lorentz_3p_v(velocity_dict[line], x0_e, x0_b, x0_b2, sigma_e, sigma_b, sigma_b2,
                                                                 amp[0], broad_amp[0], broad_amp[1])
                    else:
                        residual_dict[line] = residual_3p_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                        best_model_n_dict[line] = gaussian_1p_v(velocity_dict[line], x0_e, sigma_e, amp[0])
                        best_model_b_dict[line] = gaussian_1p_v(velocity_dict[line], x0_b, sigma_b, broad_amp[0])
                        best_model_b2_dict[line] = gaussian_1p_v(velocity_dict[line], x0_b2, sigma_b2, broad_amp[1])
                        model_dict[line] = gaussian_3p_v(velocity_dict[line], x0_e, x0_b, x0_b2, sigma_e, sigma_b, sigma_b2,
                                                         amp[0], broad_amp[0], broad_amp[1])
            # single line profile with emission+absorption components
            if (line in absorption_lines) and (line not in broad_wings_lines):
                abs_amp = [amps_dict[f"{line.split(' ')[1]}_abs"]]
                params_line = [x0_e, x0_a, sigma_e, sigma_a] + amp + abs_amp
                residual_dict[line] = residual_2p_gl_v_c(params_line, velocity_dict[line], flux_v_dict[line], err_v_dict[line])
                best_model_em_dict[line] = gaussian_1p_v(velocity_dict[line], x0_e, sigma_e, amp[0])   
                best_model_ab_dict[line] = lorentzian_1p_v(velocity_dict[line], x0_a, sigma_a, abs_amp[0])
                model_dict[line] = gaussian_lorentz_2p_v(velocity_dict[line], x0_e, x0_a, sigma_e, sigma_a, 
                                                         amp[0], abs_amp[0])
        # append the line chi2 to the chi2 dict
        ndata = len(residual_dict[line]) # number of data points
        nfixs_amp = num_fix_amps(line, amps_fixed_names) # number of fixed amplitude variables in fit
        nfixs_vel = num_fix_vels(line, fitting_method, vel_fixed_num, broad_wings_lines, 
                                 triple_gauss_lines, absorption_lines) # number of fixed velocity variables in fit
        nvarys = len(params_line) - nfixs_amp - nfixs_vel # number of variables in fit
        dof = ndata - nvarys # degree of freedom
        chisqr = np.sum(residual_dict[line]**2)
        redchi2_dict[line] = chisqr / dof
    
    return model_dict, residual_dict, best_model_n_dict, best_model_b_dict, best_model_b2_dict, best_model_em_dict, best_model_ab_dict, redchi2_dict

############################### main section for replotting ###############################
# define the absolute path of the current working directory
current_direc = os.getcwd() 

# vacuum or air wavelength
vac_or_air = 'air'

# polynomial order of continuum fitting
fit_cont_order = 1

# redshift of each galaxy
redshifts = np.array([0.01287])

# input fits name of each galaxy
fits_names = ['j1044+0353_addALL_icubes_wn']

# folder name of each galaxy
folder_names = ['j1044+0353_addALL_icubes_wn'] 

# file name of each galaxy
file_names = ['j1044+0353_addALL_icubes_wn_test']

# input text name of each galaxy
input_txt_names = ['line_selection_example']

# input fits path
input_fits_path = '../example_inputs'
# parameter table path
parameter_table_path = '../parameter_tables'
# input text file path
input_txt_path = "../input_txt"
# continuum region directory path
cont_dir_path = '../cont_dir'
# line mask region directory path
lmsk_dir_path = '../lmsk_dir'

## Plot Styling
rc('text', usetex = True)
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

# iterate over each galaxy's folder
for i, folder in enumerate(folder_names):

    # obtain the input fits file and its wavelength, flux, and error arrays
    data_fits = os.path.join(input_fits_path, fits_names[i] + '.fits')
    err_fits = os.path.join(input_fits_path, fits_names[i] + '_err.fits')
    # get the wave, sky-subtracted flux, and error arrays
    spec = fits.open(data_fits)[0].data # flux array
    espec = fits.open(err_fits)[0].data # error array
    wave = np.load(os.path.join(input_fits_path, 'wave_grid.npy')) # wavelength array

    # obtain the galaxy name based on the folder name 
    galaxy_name = folder.split('_')[2]

    # redshift of this galaxy
    redshift = redshifts[i]

    # initialize the velocity, flux, and flux_err dictionaries
    velocity_dict = dict()
    flux_v_dict = dict()
    err_v_dict = dict()
    velocity_c_dict = dict() # unmasked version
    flux_v_c_dict = dict() # unmasked version (no continuum subtracted)
    err_v_c_dict = dict() # unmasked version (no continuum subtracted)
    # initialize the line-continuum dict for all intended lines
    cont_line_dict = dict()
    # initialize the line mask dict for all masked lines
    lmsks_dict = dict()

    # obtain the selected line names 
    input_txt = os.path.join(input_txt_path, input_txt_names[i] + '.txt')
    line_select_pars, amps_ratio_dict = extract_line_pars(input_txt, return_amps_fixed_dict = True)
    fitting_method = line_select_pars['fitting_method'] # fitting method
    selected_lines = line_select_pars['selected_lines'] # selected lines 
    broad_wings_lines = line_select_pars['multi_emis_lines'] # multi-component lines
    double_gauss_lines = line_select_pars['double_gauss_lines'] # two-component lines
    triple_gauss_lines = line_select_pars['triple_gauss_lines'] # three-component lines
    absorption_lines = line_select_pars['absorption_lines'] # absorption lines
    # lines that need a Lorentzian function for fitting broad wings ("core" is fitted by Gaussians by default)
    fit_func_choices = {line: "Lorentzian" for line in line_select_pars['lorentz_bw_lines']}

    # return the fixed line component names included in self.amps_ratio_dict 
    # since each key name in self.amps_ratio_dict is in the format of A_over_B, the fixed parameter for each key name is A here (for this example).
    amps_fixed_names = [part for ratio, value in amps_ratio_dict.items() for part in extract_key_parts_from_ratio(ratio)][::2] # [::2]/[1::2] to get the even/odd index; 

    # Iterate through each selected line and return the corresponding extracted velocity, flux, and error arrays
    for indx, line in enumerate(selected_lines):
        if '&' in line:  # Special case for multilet
            multilet_lines = split_multilet_line(line)
            # doublet
            if len(multilet_lines) == 2:
                result_fit, cont_return, lmasks = extract_fit_window(wave, spec, espec, line, redshift, folder, fit_cont_order = fit_cont_order, vac_or_air = vac_or_air)
                v_arr, v_arr_2, flux_v_arr, err_v_arr, v_c_arr, v_c_arr_2, flux_v_c_arr, err_v_c_arr = result_fit
                velocity_dict[line] = np.array([v_arr, v_arr_2])
                velocity_c_dict[line] = np.array([v_c_arr, v_c_arr_2])
            # triplet
            if len(multilet_lines) == 3:
                result_fit, cont_return, lmasks = extract_fit_window(wave, spec, espec, line, redshift, folder, fit_cont_order = fit_cont_order, vac_or_air = vac_or_air)
                v_arr, v_arr_2, v_arr_3, flux_v_arr, err_v_arr, v_c_arr, v_c_arr_2, v_c_arr_3, flux_v_c_arr, err_v_c_arr = result_fit
                velocity_dict[line] = np.array([v_arr, v_arr_2, v_arr_3])
                velocity_c_dict[line] = np.array([v_c_arr, v_c_arr_2, v_c_arr_3])
        else:  # General case
            result_fit, cont_return, lmasks = extract_fit_window(wave, spec, espec, line, redshift, folder, fit_cont_order = fit_cont_order, vac_or_air = vac_or_air)
            v_arr, flux_v_arr, err_v_arr, v_c_arr, flux_v_c_arr, err_v_c_arr = result_fit
            velocity_dict[line] = v_arr
            velocity_c_dict[line] = v_c_arr
        flux_v_dict[line] = flux_v_arr
        err_v_dict[line] = err_v_arr
        flux_v_c_dict[line] = flux_v_c_arr
        err_v_c_dict[line] = err_v_c_arr
        cont_line_dict[line] = cont_return
        lmsks_dict[line] = lmasks

    # Get the best-fitting parameters for this galaxy
    folder_name_par_path = os.path.join(parameter_table_path, folder)
    parameter_file = os.path.join(folder_name_par_path, file_names[i] + '_parameters.csv')
    df = pd.read_csv(parameter_file)

    # velocity shift
    x0_e = df['velocity center'].iloc[0]
    # x0_e_err = df['velocity center'].iloc[1]
    x0_b = df['velocity center'].iloc[2]
    # x0_b_err = df['velocity center'].iloc[3]
    x0_b2 = df['velocity center'].iloc[4]
    # x0_b2_err = df['velocity center'].iloc[5]
    x0_a = df['velocity center'].iloc[6]
    # x0_a_err = df['velocity center'].iloc[7]
    x0_dict = {'x0_e': x0_e, 'x0_b': x0_b, 'x0_b2': x0_b2, 'x0_a': x0_a}

    # velocity width 
    sigma_e = df['velocity width'].iloc[8]
    # sigma_e_err = df['velocity width'].iloc[9]
    sigma_b = df['velocity width'].iloc[10]
    # sigma_b_err = df['velocity width'].iloc[11]
    sigma_b2 = df['velocity width'].iloc[12]
    # sigma_b2_err = df['velocity width'].iloc[13]
    sigma_a = df['velocity width'].iloc[14]
    # sigma_a_err = df['velocity width'].iloc[15]
    sigma_dict = {'sigma_e': sigma_e, 'sigma_b': sigma_b, 'sigma_b2': sigma_b2, 'sigma_a': sigma_a}

    # line amplitudes
    amps_all = df['line amplitude'].iloc[16:]
    # extract the corresponding row names (keys) for these values
    row_names = df['Unnamed: 0'].iloc[16:]
    # create a dictionary with keys as requested, removing "amp_" prefix
    amps_dict = {row_name.replace('amp_', ''): amp for row_name, amp in zip(row_names, amps_all)}
 
    # return dictionaries for best-fitting results
    model_dict, residual_dict, best_model_n_dict, best_model_b_dict, best_model_b2_dict, best_model_em_dict, \
    best_model_ab_dict, redchi2_dict = return_best_models(line, fitting_method, broad_wings_lines, double_gauss_lines, triple_gauss_lines, 
                                                          fit_func_choices, absorption_lines, x0_dict, sigma_dict, amps_dict, amps_fixed_names)

    ##################################################################### create the plotting figure
    # determine the number of plots based on the number of selected lines

    if vac_or_air == 'air':
        elements_dict = read_wavelengths_from_file('../doc/air_wavelengths.txt', redshift)
    if vac_or_air == 'vac':
        elements_dict = read_wavelengths_from_file('../doc/vac_wavelengths.txt', redshift)
    wave_dict = merge_element_waves(elements_dict)

    num_plots = len(selected_lines)

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
    cont_line_v_dict = dict()
    flux_plus_cont_v_dict = dict()
    flux_plus_cont_v_dict_2 = dict()
    model_plus_cont_v_dict = dict()
    # whether include broad_wings_lines
    if broad_wings_lines:
        best_model_plus_cont_n_v_dict = dict()
        best_model_plus_cont_b_v_dict = dict()
        if triple_gauss_lines:
            best_model_plus_cont_b2_v_dict = dict()
    # whether include absorption_lines
    if absorption_lines:
        best_model_plus_cont_em_v_dict = dict()
        best_model_plus_cont_ab_v_dict = dict()

    for line in selected_lines:
        # Choose the appropriate line wave based on the length of line_waves
        line_waves = wave_dict[line]
        chosen_line_wave = line_waves[0] if len(line_waves) <= 2 else line_waves[1]
        # multilet that needs to be fitted together
        if '&' in line:  # Special case for doublet that should be fitted together
            multilet_lines = split_multilet_line(line)
            cont_line_v_dict[line] = cont_line_dict[line][0] * chosen_line_wave / c
            for ii, l in enumerate(multilet_lines):
                model_plus_cont_v_dict[l] = model_dict[l] + cont_line_v_dict[line]
                if l in broad_wings_lines:
                    best_model_plus_cont_n_v_dict[l] = best_model_n_dict[l] + cont_line_v_dict[line]
                    best_model_plus_cont_b_v_dict[l] = best_model_b_dict[l] + cont_line_v_dict[line]
                    if l in triple_gauss_lines:
                        best_model_plus_cont_b2_v_dict[l] = best_model_b2_dict[l] + cont_line_v_dict[line]
        # single line profile
        else: # General case
            cont_line_v_dict[line] = cont_line_dict[line][0] * chosen_line_wave / c
        flux_plus_cont_v_dict[line] = flux_v_c_dict[line]
        flux_plus_cont_v_dict_2[line] = flux_v_dict[line] + cont_line_v_dict[line]
        model_plus_cont_v_dict[line] = model_dict[line] + cont_line_v_dict[line]
        # whether include broad_wings_lines
        if line in broad_wings_lines:
            best_model_plus_cont_n_v_dict[line] = best_model_n_dict[line] + cont_line_v_dict[line]
            best_model_plus_cont_b_v_dict[line] = best_model_b_dict[line] + cont_line_v_dict[line]
            if line in triple_gauss_lines:
                best_model_plus_cont_b2_v_dict[line] = best_model_b2_dict[line] + cont_line_v_dict[line]
        # whether include absorption_lines
        if line in absorption_lines:
            best_model_plus_cont_em_v_dict[line] = best_model_em_dict[line] + cont_line_v_dict[line]
            best_model_plus_cont_ab_v_dict[line] = best_model_ab_dict[line] + cont_line_v_dict[line]

    # plot the fitting results
    for i, line in enumerate(selected_lines):
        # Choose the appropriate line wave based on the length of line_waves
        line_waves = wave_dict[line]
        chosen_line_wave = line_waves[0] if len(line_waves) <= 2 else line_waves[1]
        if '&' in line:
            multilet_lines = split_multilet_line(line)
            line_name = '&'.join(multilet_lines[:min(len(multilet_lines), 3)])
            v_arr = velocity_dict[line][0] if len(multilet_lines) == 2 else velocity_dict[line][1]
            v_c_arr = velocity_c_dict[line][0] if len(multilet_lines) == 2 else velocity_c_dict[line][1]
            for l in multilet_lines:
                if l in broad_wings_lines:
                    axes[0,i].plot(v_arr, best_model_plus_cont_n_v_dict[l], 'c--',
                                   zorder = 3, lw = 2)
                    axes[0,i].plot(v_arr, best_model_plus_cont_b_v_dict[l], 'b--',
                                   zorder = 3, lw = 2)
                    if l in triple_gauss_lines:
                        axes[0,i].plot(v_arr, best_model_plus_cont_b2_v_dict[l], 'g--',
                                       zorder = 3, lw = 2)
        else:
            v_arr = velocity_dict[line]
            v_c_arr = velocity_c_dict[line]
            line_name = line
        # upper panel for plotting the raw and best-fitting line profile
        axes[0,i].step(v_c_arr, flux_plus_cont_v_dict[line], 'k', where = 'mid')
        # axes[0,i].fill_between(v_c_arr, (flux_plus_cont_v_dict[line]+err_v_c_dict[line]) / np.max(flux_plus_cont_v_dict[line]),
        #                       (flux_plus_cont_v_dict[line]-err_v_c_dict[line]) / np.max(flux_plus_cont_v_dict[line]), alpha =0.5, zorder = 1,
        #                        facecolor = 'black')
        axes[0,i].fill_between(v_arr, (flux_plus_cont_v_dict_2[line]-err_v_dict[line]),
                              (flux_plus_cont_v_dict_2[line]+err_v_dict[line]), alpha =0.5, zorder = 1,
                               facecolor = 'black')
        axes[0,i].plot(v_arr, model_plus_cont_v_dict[line], 'r--', zorder = 2, lw = 2)
        if line in broad_wings_lines:
            axes[0,i].plot(v_arr, best_model_plus_cont_n_v_dict[line], 'c--',
                           zorder = 3, lw = 2)
            axes[0,i].plot(v_arr, best_model_plus_cont_b_v_dict[line], 'b--',
                           zorder = 3, lw = 2)
            if line in triple_gauss_lines:
                axes[0,i].plot(v_arr, best_model_plus_cont_b2_v_dict[line], 'g--',
                               zorder = 3, lw = 2)
        if line in absorption_lines:
            axes[0,i].plot(v_arr, best_model_plus_cont_em_v_dict[line], 'c--',
                           zorder = 2, lw = 2)
            axes[0,i].plot(v_arr, best_model_plus_cont_ab_v_dict[line], 'b--',
                           zorder = 2, lw = 2)
        axes[0,i].set_yscale('log')

        # axes[0,i].text(0.04, 0.92, line_name.replace('&', ' \& '), size = 18, transform=axes[0,i].transAxes, va="center",color="black")
        axes[0,i].text(0.04, 0.85, line_name.replace('&', ' \& ') + '\n' + r'$\chi^2 = $' + "{0:.2f}".format(redchi2_dict[line]), 
                       size = 18, transform=axes[0,i].transAxes, va="center",color="black")
        # axes[0,i].axvline(x = 0, ls = '--', color = 'grey', lw = 2) # might be confusing for multiplet
        axes[0,i].tick_params(axis='y', which='minor')
        axes[0,i].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if i == 0:
            axes[0,i].set_ylabel(r'Flux ($10^{-17} \ \rm{erg \ s^{-1} \ cm^{-2} \ (km \ s^{-1})^{-1}}$)',size = 18)
            # axes[0,i].set_ylabel(r'Normalized Flux',size = 22)
        ymin = 0.9 * min(flux_plus_cont_v_dict[line])
        ymax = 1.1 * max(flux_plus_cont_v_dict[line])
        axes[0,i].set_ylim(ymin, ymax)

        ##### lower panel for plotting the residual (data - model) / error
        # axes[1,i].step(v_arr, residual_dict[line], where = 'mid')
        # [axes[1,i].axhline(y=j, color="red", linestyle='--') for j in [0,-1,1]]
        # axes[1,i].set_xlabel(r'Velocity $\mathrm{(km \ s^{-1})}$',size = 22)
        # if i == 0:
        #     axes[1,i].set_ylabel(r'Residual',size = 22)
        # ymin2 = 1.05 * min(residual_dict[line])
        # ymax2 = 1.05 * max(residual_dict[line])
        # axes[1,i].set_ylim(ymin2, ymax2)
        ##### 
        ##### lower panel for plotting the residual (data - model)
        residual_flux = residual_dict[line] * err_v_dict[line]
        axes[1,i].step(v_arr, residual_flux, 'k', where = 'mid')
        axes[1,i].set_xlabel(r'Velocity $\rm{(km \ s^{-1})}$',size = 22)
        # if i == 0:
        #     axes[1,i].set_ylabel(r'Residual',size = 22)
        ymin2 = 1.1 * min(residual_flux)
        ymax2 = 1.1 * max(residual_flux)
        axes[1,i].set_ylim(ymin2, ymax2)
        ##### 
        
        ##### plot the masked regions
        try:
            lmsks = lmsks_dict[line]
            if len(lmsks) >= 1:
                for ml in lmsks:
                    x_lmsk = np.linspace(ml['w0'], ml['w1'], 100)
                    v_lmsk = (x_lmsk / chosen_line_wave - 1) * c
                    axes[0,i].fill_between(v_lmsk, ymin, ymax, alpha=0.3, zorder=1, facecolor='orange')
                    axes[1,i].fill_between(v_lmsk, ymin2, ymax2, alpha=0.3, zorder=1, facecolor='orange')
        except KeyError:
            print(f"\nno masked regions to plot for {line}.")
        ##### 
    # define the current working directory
    current_direc = os.getcwd()
    # define the results directory based on the sub-folder name
    results_dir = os.path.join(current_direc, f"replots/{folder}/")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # define the filename and save it in the sub-directory output_files
    filepath = os.path.join(results_dir, file_names[i] + '_fittings.pdf')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.clf() # not showing the matplotlib figure; check the "plots" folder
    #####################################################################

############################### main section for replotting ###############################






