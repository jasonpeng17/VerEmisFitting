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

# redshift of the galaxy
redshift = 0.01287
# whether the spectrum is in "vacuum" or "air" wavelength space
vac_or_air = 'air'

# run the fitting result for each selected line profile
# "n_iteration = 1000" defines the number of iterations you want to run
# "get_flux = True" defines if you want the return to be the flux dict (includes the flux of each line profile) or not; if False, then the return is the best-fitting parameters
# "get_error = True" defines if you want to calculate the error of each line flux 
# "get_corr = True" defines if you want the flux to be extinction-corrected or not
# "get_ew = True" defines if you want to calculate the ew(s) of the selected emission lines (including emission and absorption ew(s))
# "save_flux_table = True" defines if you want to save the best-fitting flux pandas table for each line.
# "save_ew_table = True" defines if you want to save the best-fitting equivalent width pandas table for each line.
# "save_sigma_table = True" defines if you want to save the best-fitting velocity width pandas table for each velocity component.
# "line_selection_method = 'gui' or 'txt' defines how you want to select the lines for fitting, whether using a GUI or inputting a txt"
input_example_txt = current_direc + '/input_txt/line_selection_example.txt'
region = line_fitting_exec(redshift = redshift, vac_or_air = vac_or_air, E_BV = None, fits_name = data_fits, line_select_method = 'gui', input_txt = input_example_txt)
region.all_lines_result(wave, spec, err, n_iteration = 1000, get_flux = True, get_corr = False, save_flux_table = True, 
                        get_ew = True, save_ew_table = True, get_error = True, save_par_table = True)

# plot the fitting result
# "savefig = True" defines if you want to save the fitting result as a .pdf file.
region.fitting_plot(savefig = True)













