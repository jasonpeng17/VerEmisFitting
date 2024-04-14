# this file is for running the line-fitting models and plot the best-fitting results 
import numpy as np
from astropy.io import fits
import os, sys
from line_fitting_exec import *
from line_fitting_model import *

# define the absolute path of the current working directory
current_direc = os.getcwd() 

# # define the absolute path of the input fits files
data_fits = current_direc + '/example_inputs/j1044+0353_addALL_icubes_wn.fits'
err_fits = current_direc + '/example_inputs/j1044+0353_addALL_icubes_wn_err.fits'
spec = fits.open(data_fits)[0].data # flux array
err = fits.open(err_fits)[0].data # error array
wave = np.load(current_direc + '/example_inputs/wave_grid.npy') # wavelength array

############################### INPUT PARAMETERS (run the fitting result for each selected line profile) 
# redshift of the galaxy
redshift = 0.01287
# whether the spectrum is in "vac" or "air" wavelength space
vac_or_air = 'air'
# the order of fitting local continuum level
fit_cont_order = 1
# "line_selection_method = 'gui' or 'txt' defines how you want to select the lines for fitting, whether using a GUI or inputting a txt"
line_select_method = 'gui'
# text file for selecting intended lines for fitting
input_example_txt = current_direc + '/input_txt/line_selection_example.txt' 
# whether to interactively determine the fitting window, local continuum regions, and masking lines 
fit_window_gui = True # if False, use the default values 
# whether to pop up the GUI window for interactively determine the initial parameter values and their corresponding ranges (for each iteration). Default is False (i.e., pop up the window)
params_windows_gui = True # if False, use the default parameter initial values and corresponding ranges for each iteration
# define the folder name 
folder_name = data_fits.split('/')[-1][:-5] # if None, then a tk window will pop up for users to interactively enter the folder name; if users forget to type the folder name in the tk window,
                                            # the default folder name is "test_folder".
# define the file name
file_name = "test_2" # if None, then file_name = folder_name; else, file_name will be f"{folder_name}_{file_name}"
# define the minimum velocity width (i.e., instrumental seeing) for each velocity component
sigma_min = 30 # in km / s
# define the maximum velocity width for each velocity component (for emissions)
sigma_max_e = 1200 # in km / s
# define the maximum velocity width for each velocity component (for absorptions)
sigma_max_a = 1500 # in km / s
# define the algorithm used for fitting (the well-tested ones include ’leastsq’: Levenberg-Marquardt (default), ’nelder’: Nelder-Mead, and ’powell’: Powell)
fit_algorithm = "leastsq"
###############################
region = line_fitting_exec(redshift = redshift, vac_or_air = vac_or_air, folder_name = folder_name, file_name = file_name, line_select_method = line_select_method, 
                           input_txt = input_example_txt, fit_cont_order = fit_cont_order, fit_window_gui = fit_window_gui, params_windows_gui = params_windows_gui,
                           sigma_min = sigma_min, sigma_max_e = sigma_max_e, sigma_max_a = sigma_max_a, fit_algorithm = fit_algorithm)

# "n_iteration = 1000" defines the number of iterations you want to run
# "get_flux = True" defines if you want the return to be the flux dict (includes the flux of each line profile) or not; if False, then the return is the best-fitting parameters
# "get_error = True" defines if you want to calculate the error of each line flux 
# "get_ew = True" defines if you want to calculate the ew(s) of the selected emission lines (including emission and absorption ew(s))
# "save_flux_table = True" defines if you want to save the best-fitting flux pandas table for each line.
# "save_ew_table = True" defines if you want to save the best-fitting equivalent width pandas table for each line.
# "save_par_table = True" defines if you want to save the best-fitting parameter pandas table for each velocity component.
# "save_stats_table = True" defines if you want to save the best-fitting statistics pandas table for each selected line.
# "save_cont_params_table = True" defines if you want to save the best-fitting parameters for continuum fit for each line.
region.all_lines_result(wave, spec, err, n_iteration = 100, get_flux = True, save_flux_table = True, get_ew = True, save_ew_table = True, get_error = True, 
                        save_par_table = True, save_stats_table = True, save_cont_params_table = True)

# plot the fitting result
# "savefig = True" defines if you want to save the fitting result as a .pdf file.
region.fitting_plot(savefig = True)
