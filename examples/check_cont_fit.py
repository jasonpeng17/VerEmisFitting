# This file is for users to check the fitted continuum levels and plot their own best-fitting figures.

import numpy as np
from astropy.io import fits
import os, sys
from line_fitting_exec import *
from line_fitting_model import *

#################### Copy from run_line_fitting.py
# define the absolute path of the current working directory
current_direc = os.getcwd() 

# # define the absolute path of the input fits files
data_fits = current_direc + '/example_inputs/j1044+0353_addALL_icubes_wn.fits'
err_fits = current_direc + '/example_inputs/j1044+0353_addALL_icubes_wn_err.fits'
spec = fits.open(data_fits)[0].data # flux array
err = fits.open(err_fits)[0].data # error array
wave = np.load(current_direc + '/example_inputs/wave_grid.npy') # wavelength array

# run the fitting result for each selected line profile
# redshift of the galaxy
redshift = 0.01287
# whether the spectrum is in "vacuum" or "air" wavelength space
vac_or_air = 'air'
# the order of fitting local continuum level
fit_cont_order = 1
# "line_selection_method = 'gui' or 'txt' defines how you want to select the lines for fitting, whether using a GUI or inputting a txt"
input_example_txt = current_direc + '/input_txt/line_selection_example.txt' # text file for selecting intended lines for fitting
# whether to interactively determine the fitting window, local continuum regions, and masking lines 
fit_window_gui = True # if False, use the default values 
fit_result = line_fitting_exec(redshift = redshift, vac_or_air = vac_or_air, fits_name = data_fits, line_select_method = 'gui', 
                               input_txt = input_example_txt, fit_cont_order = fit_cont_order, fit_window_gui = fit_window_gui)

# "n_iteration = 1000" defines the number of iterations you want to run
# "get_flux = True" defines if you want the return to be the flux dict (includes the flux of each line profile) or not; if False, then the return is the best-fitting parameters
# "get_error = True" defines if you want to calculate the error of each line flux 
# "get_ew = True" defines if you want to calculate the ew(s) of the selected emission lines (including emission and absorption ew(s))
# "save_flux_table = True" defines if you want to save the best-fitting flux pandas table for each line.
# "save_ew_table = True" defines if you want to save the best-fitting equivalent width pandas table for each line.
# "save_sigma_table = True" defines if you want to save the best-fitting velocity width pandas table for each velocity component.
fit_result.all_lines_result(wave, spec, err, n_iteration = 1000, get_flux = True, save_flux_table = True, get_ew = True, save_ew_table = True, get_error = True, save_par_table = True)

# plot the fitting result
# "savefig = True" defines if you want to save the fitting result as a .pdf file.
fit_result.fitting_plot(savefig = True)
#################### 

# Step (1): check the fitting window and the local continuum regions in the cont_dir folder
# The fitting window and local continuum region information will be saved to the `cont_dir` folder as a `.cont` file, specifying four boundary points in Angstroms: `x1` (lower-end of the fitting window), 
# `x2` (upper-end of the fitting window), `x3` (upper-end of the left local continuum region), and `x4` (lower-end of the right local continuum region). 
# These define the local continuum regions asx1 - x3 and x4 - x2.

# Step (2): check the masked regions in the lmsk_dir folder
# Each line (with two boundary points in Angstroms) defining a region to be masked during fitting.

# Step (3): print the fitted continuum level directory (contains the fitted continuum level for each line)
print(fit_result.cont_line_dict)

# Step (4): check the best-fitting parameter values in the parameter_tables folder.

# Step (5): plot your own best-fitting figures!






