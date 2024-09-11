# Zixuan Peng modified on 08/12/2024 to push to the VerEmisFitting github repository
# Yuan Li created on 03/06/2024 to test on J0944 bins

# Import necessary libraries
import numpy as np
from astropy.io import fits
import os, sys
from astropy.wcs import WCS
import shutil

# Add the parent directory to sys.path for importing custom modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

# Import custom line fitting modules
from veremisfitting.line_fitting_exec import *
from veremisfitting.line_fitting_model import *

from IPython import embed  # For debugging purposes

##################### Step (1): Path Definitions
# Notes: the voronoi binned fits files are generated by the Python package MVT-binning (https://github.com/pierrethx/MVT-binning)

# Define paths to necessary directories and files
vef_dir = '/Applications/VerEmisFitting' # VerEmisFitting folder
input_spec_dir = '/Volumes/Seagate Portable Drive/kcwi_velmap/j0944' # parent directory of j0944 voronoi bins 
bin_fits = input_spec_dir + '/binning_v3/target20_WVT2s/z_image.j0944-0038_pa-30_combined_icubes5006_29_WVT2s_assigned.fits'

# Additional FITS files for signal and data
bin_signal_fits = input_spec_dir + '/binning_v3/target20_WVT2s/block_image.j0944-0038_pa-30_combined_icubes5006_29_WVT2s_sig.fits'
data_fits = input_spec_dir + '/cubes/j0944-0038_pa-30_combined_icubes.fits'
err_fits = input_spec_dir + '/cubes/j0944-0038_pa-30_combined_vcubes.fits'
lmsk_file_ls = [vef_dir + '/lmsk_dir/j0944/default/[OIII]&HeI_5007&5015.lmsk']
#####################

##################### Step (2): Helper Functions
# Several helper functions (get_folder_name, get_file_name, get_patch_centroid_xy, get_patch_xyrA, get_xyrA_string, get_wave_axis) 
# are defined to streamline repetitive tasks like generating folder/file names, calculating centroids, and creating a wavelength axis.

# Function to get the folder name based on the bin number and XYR (x, y, r, Area) string
def get_folder_name(xyrA_string=None):
    if xyrA_string:
        return f'j0944/binned/bin{int(bin_num)}_{xyrA_string}'
    else:
        return f'j0944/binned/bin{int(bin_num)}'

# Function to get the file name for the output
def get_file_name():
    return 'fit_v1'

# Function to get the centroid of a specific patch in an image
def get_patch_centroid_xy(image, value):
    '''
    returns the x,y value of the centroid of all pixels in an image that equals to value
    '''
    y_indices, x_indices = np.where(image == value)
    x_mean, y_mean = np.mean(x_indices), np.mean(y_indices)
    return x_mean, y_mean

# Function to calculate x, y, r, and A (Area) for a patch in the image
def get_patch_xyrA(seg_img, bin_num, bin_signal_image):
    '''
    1. find the centroid of brightest bin in bin_signal_image
    2. find the centroid of the seg_img of bin_num
    3. subtract the 2 centroids to find x, y
    4. r=sqrt(x^2+y^2)
    5. A is the area of the patch
    '''

    brightest_bin_value = np.max(bin_signal_image)
    x_mean_brightest, y_mean_brightest = get_patch_centroid_xy(bin_signal_image, brightest_bin_value)

    x_mean_seg, y_mean_seg = get_patch_centroid_xy(seg_img, bin_num)

    x = x_mean_seg - x_mean_brightest
    y = y_mean_seg - y_mean_brightest

    r = np.sqrt(x**2 + y**2)

    A = np.sum(np.where(seg_img == bin_num, 1, 0))

    return x, y, r, A

# Function to generate a string representing x, y, r, and A
def get_xyrA_string(x, y, r, A):
    return f'x={x:.1f}_y={y:.1f}_r={r:.1f}_A={A:.0f}'
#####################


##################### Step (3): Define Fitting Parameters
# Define fitting parameters
redshift = 0.004776 # redshift of the galaxy
vac_or_air = 'air' # whether the spectrum is in "vac" or "air" wavelength space
fit_cont_order = 1 # the polynomial order of fitting local continuum level
line_select_method = 'txt' # "line_selection_method = 'gui' or 'txt' defines how you want to select the lines for fitting, whether using a GUI or inputting a txt"
input_example_txt = vef_dir + '/input_txt/OIII49595007HeI5015_3triple_lorentz.txt'
fit_window_gui = False  # if False, use the default values 
params_windows_gui = False  # if False, use the default parameter initial values and corresponding ranges for each iteration
# Define the minimum velocity width (i.e., instrumental seeing) for each velocity component
sigma_min = 30 # in km / s
# Define the maximum velocity width for each velocity component (for emissions)
sigma_max_e = 1200 # in km / s
# Define the maximum velocity width for each velocity component (for absorptions)
sigma_max_a = 1500 # in km / s
# Define the algorithm used for fitting (the well-tested ones include ’leastsq’: Levenberg-Marquardt (default), ’nelder’: Nelder-Mead, and ’powell’: Powell)
fit_algorithm = "leastsq"  # Algorithm used for fitting
##################### 

##################### Step (4): Data Reading
# Read data and variance cubes
data_arr = fits.open(data_fits)[0].data
err_arr = fits.open(err_fits)[0].data
wcs = WCS(fits.open(data_fits)[0].header)

# Function to create the wavelength axis
def get_wave_axis(data_arr, wcs):
    wlaxis = np.arange(data_arr.shape[0])
    laxis = wcs.all_pix2world(1, 1, wlaxis, 0)[2] * 1e10
    return laxis

# Generate wavelength axis
wave = get_wave_axis(data_arr, wcs)

# Read bin_fits to set up the loop over bins
seg_img = fits.open(bin_fits)[0].data
bin_num_ls = np.unique(seg_img)

# Read bin_signal_fits to find the center of the brightest bin
bin_signal_image = fits.open(bin_signal_fits)[0].data
##################### 


##################### Step (5): Main Loop Over Bins
# The script loops over each bin in the segmented image, performing the following steps:
# (1) Extracts the spectrum and error for the bin.
# (2) Determines the centroid and area of the bin.
# (3) Copies the line mask files to a new directory specific to this bin.
# (4) Executes the line fitting process using the line_fitting_exec function from the veremisfitting module.
# (5) Saves the results, including flux tables, equivalent width (EW) tables, parameter tables, and plots.

# Loop over each bin in the segmented image
for bin_num in bin_num_ls:
    # Extract flux by summing over all pixels in the bin
    spec = np.sum(np.where(seg_img == bin_num, data_arr, 0), axis=(1, 2))
    
    # Calculate error by summing variance and then taking the square root
    err = np.sqrt(np.sum(np.where(seg_img == bin_num, err_arr, 0), axis=(1, 2)))

    # Save the lmsk file (line mask file) for this bin
    x, y, r, A = get_patch_xyrA(seg_img, bin_num, bin_signal_image)
    xyrA_string = get_xyrA_string(x, y, r, A)
    
    lmsk_dir = f"lmsk_dir/{get_folder_name(xyrA_string=xyrA_string)}"
    if not os.path.exists(lmsk_dir):
        os.makedirs(lmsk_dir)
    for lmsk_file in lmsk_file_ls:
        shutil.copy(lmsk_file, lmsk_dir)

    # Set folder and file names for output
    folder_name = get_folder_name(xyrA_string=xyrA_string)
    file_name = get_file_name()

    # Execute the line fitting using the VerEmisFitting tool
    region = line_fitting_exec(redshift = redshift, vac_or_air = vac_or_air, folder_name = folder_name, file_name = file_name, line_select_method = line_select_method, 
                               input_txt = input_example_txt, fit_cont_order = fit_cont_order, fit_window_gui = fit_window_gui, params_windows_gui = params_windows_gui,
                               sigma_min = sigma_min, sigma_max_e = sigma_max_e, sigma_max_a = sigma_max_a, fit_algorithm = fit_algorithm)

    # Run the fitting process and save results
    region.all_lines_result(wave, spec, err, n_iteration = 1000, get_flux = True, save_flux_table = True, get_ew = True, save_ew_table = True, get_error = True, 
                            save_par_table = True, save_stats_table = True, save_cont_params_table = True)

    # Save the fitting plot
    region.fitting_plot(savefig=True)
#####################





