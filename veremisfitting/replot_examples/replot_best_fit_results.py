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
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from veremisfitting.fitting_window_gui import *
from veremisfitting.analysis_utils import *
from veremisfitting.data_loading_utils import *
from veremisfitting.modeling_utils import *
from veremisfitting.line_fitting_model import *
from veremisfitting.line_fitting_exec import *

############################### main section for running line-fitting process ###############################
# # define the absolute path of the input fits files
data_fits = parent_dir + '/example_inputs/j1044+0353_addALL_icubes_wn.fits'
err_fits = parent_dir + '/example_inputs/j1044+0353_addALL_icubes_wn_err.fits'
spec = fits.open(data_fits)[0].data # flux array
err = fits.open(err_fits)[0].data # error array
wave = np.load(parent_dir + '/example_inputs/wave_grid.npy') # wavelength array

############################### INPUT PARAMETERS (run the fitting result for each selected line profile) 
# redshift of the galaxy
redshift = 0.01287
# whether the spectrum is in "vac" or "air" wavelength space
vac_or_air = 'air'
# the polynomial order of fitting local continuum level
fit_cont_order = 1
# "line_selection_method = 'gui' or 'txt' defines how you want to select the lines for fitting, whether using a GUI or inputting a txt"
line_select_method = 'txt'
# text file for selecting intended lines for fitting
input_example_txt = parent_dir + '/input_txt/line_selection_example.txt' 
# whether to interactively determine the fitting window, local continuum regions, and masking lines 
fit_window_gui = False # if False, use the default values 
# whether to pop up the GUI window for interactively determine the initial parameter values and their corresponding ranges (for each iteration). Default is False (i.e., pop up the window)
params_windows_gui = False # if False, use the default parameter initial values and corresponding ranges for each iteration
# define the folder name 
folder_name = data_fits.split('/')[-1][:-5] # if None, then a tk window will pop up for users to interactively enter the folder name; if users forget to type the folder name in the tk window,
                                            # the default folder name is "test_folder".
# define the file name
file_name = "test" # if None, then file_name = folder_name; else, file_name will be f"{folder_name}_{file_name}"
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
region.all_lines_result(wave, spec, err, n_iteration = 100, get_flux = True, save_flux_table = False, get_ew = True, save_ew_table = False, get_error = True, 
                        save_par_table = False, save_stats_table = False, save_cont_params_table =False)
############################### main section for running line-fitting process ###############################

############################### main section for replotting ###############################
# determine the number of plots based on the number of selected lines
num_plots = len(region.selected_lines)

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
cont_line_v_dict = dict()
flux_plus_cont_v_dict = dict()
model_plus_cont_v_dict = dict()
# whether include broad_wings_lines
if region.broad_wings_lines:
    best_model_plus_cont_n_v_dict = dict()
    best_model_plus_cont_b_v_dict = dict()
    if region.triple_gauss_lines:
        best_model_plus_cont_b2_v_dict = dict()
# whether include absorption_lines
if region.absorption_lines:
    best_model_plus_cont_ab_v_dict = dict()

for line in region.selected_lines:
    # Choose the appropriate line wave based on the length of line_waves
    line_waves = region.wave_dict[line]
    chosen_line_wave = line_waves[0] if len(line_waves) <= 2 else line_waves[1]
    # multilet that needs to be fitted together
    if '&' in line:  # Special case for doublet that should be fitted together
        multilet_lines = split_multilet_line(line)
        cont_line_v_dict[line] = region.cont_line_dict[line][0] * chosen_line_wave / region.c
        for ii, l in enumerate(multilet_lines):
            model_plus_cont_v_dict[l] = region.model_dict[l] + cont_line_v_dict[line]
            if l in region.broad_wings_lines:
                best_model_plus_cont_n_v_dict[l] = region.best_model_n_dict[l] + cont_line_v_dict[line]
                best_model_plus_cont_b_v_dict[l] = region.best_model_b_dict[l] + cont_line_v_dict[line]
                if l in region.triple_gauss_lines:
                    best_model_plus_cont_b2_v_dict[l] = region.best_model_b2_dict[l] + cont_line_v_dict[line]
            if l in region.absorption_lines:
                best_model_plus_cont_ab_v_dict[l] = region.best_model_ab_dict[l] + cont_line_v_dict[line]
    # single line profile
    else: # General case
        cont_line_v_dict[line] = region.cont_line_dict[line][0] * chosen_line_wave / region.c
    # self.flux_plus_cont_v_dict[line] = self.flux_v_dict[line] + self.cont_line_v_dict[line]
    flux_plus_cont_v_dict[line] = region.flux_v_c_dict[line]
    model_plus_cont_v_dict[line] = region.model_dict[line] + cont_line_v_dict[line]
    # whether include broad_wings_lines
    if line in region.broad_wings_lines:
        best_model_plus_cont_n_v_dict[line] = region.best_model_n_dict[line] + cont_line_v_dict[line]
        best_model_plus_cont_b_v_dict[line] = region.best_model_b_dict[line] + cont_line_v_dict[line]
        if line in region.triple_gauss_lines:
            best_model_plus_cont_b2_v_dict[line] = region.best_model_b2_dict[line] + cont_line_v_dict[line]
    # whether include absorption_lines
    if line in region.absorption_lines:
        best_model_plus_cont_ab_v_dict[line] = region.best_model_ab_dict[line] + cont_line_v_dict[line]

# plot the fitting results
for i, line in enumerate(region.selected_lines):
    # Choose the appropriate line wave based on the length of line_waves
    line_waves = region.wave_dict[line]
    chosen_line_wave = line_waves[0] if len(line_waves) <= 2 else line_waves[1]
    if '&' in line:
        multilet_lines = split_multilet_line(line)
        line_name = '&'.join(multilet_lines[:min(len(multilet_lines), 3)])
        v_arr = region.velocity_dict[line][0] if len(multilet_lines) == 2 else region.velocity_dict[line][1]
        v_c_arr = region.velocity_c_dict[line][0] if len(multilet_lines) == 2 else region.velocity_c_dict[line][1]
        for l in multilet_lines:
            if l in region.broad_wings_lines:
                axes[0,i].plot(v_arr, best_model_plus_cont_n_v_dict[l] / np.max(flux_plus_cont_v_dict[line]), 'c--',
                               zorder = 3, lw = 2)
                axes[0,i].plot(v_arr, best_model_plus_cont_b_v_dict[l] / np.max(flux_plus_cont_v_dict[line]), 'b--',
                               zorder = 3, lw = 2)
                if l in region.triple_gauss_lines:
                    axes[0,i].plot(v_arr, best_model_plus_cont_b2_v_dict[l] / np.max(flux_plus_cont_v_dict[line]), 'g--',
                                   zorder = 3, lw = 2)
            if l in region.absorption_lines:
                axes[0,i].plot(v_arr, best_model_plus_cont_ab_v_dict[l] / np.max(flux_plus_cont_v_dict[line]), ls = '--', color = 'orange',
                               zorder = 3, lw = 2)
    else:
        v_arr = region.velocity_dict[line]
        v_c_arr = region.velocity_c_dict[line]
        line_name = line
    # upper panel for plotting the raw and best-fitting line profile
    axes[0,i].step(v_c_arr, flux_plus_cont_v_dict[line]/np.max(flux_plus_cont_v_dict[line]), 'k', where = 'mid')
    axes[0,i].fill_between(v_c_arr, (flux_plus_cont_v_dict[line]-region.err_v_c_dict[line]) / np.max(flux_plus_cont_v_dict[line]),
                          (flux_plus_cont_v_dict[line]+region.err_v_c_dict[line]) / np.max(flux_plus_cont_v_dict[line]), alpha =0.5, zorder = 1, facecolor = 'black')
    axes[0,i].plot(v_arr, model_plus_cont_v_dict[line] / np.max(flux_plus_cont_v_dict[line]), 'r--', zorder = 2, lw = 2)
    if line in region.broad_wings_lines:
        axes[0,i].plot(v_arr, best_model_plus_cont_n_v_dict[line] / np.max(flux_plus_cont_v_dict[line]), 'c--',
                       zorder = 3, lw = 2)
        axes[0,i].plot(v_arr, best_model_plus_cont_b_v_dict[line] / np.max(flux_plus_cont_v_dict[line]), 'b--',
                       zorder = 3, lw = 2)
        if line in region.triple_gauss_lines:
            axes[0,i].plot(v_arr, best_model_plus_cont_b2_v_dict[line] / np.max(flux_plus_cont_v_dict[line]), 'g--',
                           zorder = 3, lw = 2)
    if line in region.absorption_lines:
        axes[0,i].plot(v_arr, best_model_plus_cont_ab_v_dict[line] / np.max(flux_plus_cont_v_dict[line]), ls = '--', color = 'orange',
                       zorder = 3, lw = 2)
    axes[0,i].set_yscale('log')
    axes[0,i].text(0.04, 0.92, line_name.replace('&', ' \& ') + '\n' + r'$\chi^2 = $' + "{0:.2f}".format(region.redchi2_dict[line]), 
                   size = 14, transform=axes[0,i].transAxes, va="center",color="black")
    # axes[0,i].axvline(x = 0, ls = '--', color = 'grey', lw = 2) # might be confusing for multiplet
    axes[0,i].tick_params(axis='y', which='minor')
    axes[0,i].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if i == 0:
        axes[0,i].set_ylabel(r'Normalized Flux',size = 22)
    ymin = 0.9 * min(flux_plus_cont_v_dict[line]/np.max(flux_plus_cont_v_dict[line]))
    ymax = 1.1
    axes[0,i].set_ylim(ymin, ymax)
    # axes[0,i].legend(loc='upper left', fontsize = 13, framealpha = 0)
    # lower panel for plotting the residual
    axes[1,i].step(v_arr, region.residual_dict[line], 'k', where = 'mid')
    # [axes[1,i].axhline(y=j, color="red", linestyle='--') for j in [0,-1, 1]]
    axes[1,i].set_xlabel(r'Velocity $\rm{(km \ s^{-1})}$',size = 22)
    if i == 0:
        axes[1,i].set_ylabel(r'Residual', size = 22)
    ymin2 = 1.05 * min(region.residual_dict[line])
    ymax2 = 1.05 * max(region.residual_dict[line])
    axes[1,i].set_ylim(ymin2, ymax2)
    # plot the masked regions
    try:
        lmsks = region.lmsks_dict[line]
        if len(lmsks) >= 1:
            for ml in lmsks:
                x_lmsk = np.linspace(ml['w0'], ml['w1'], 100)
                v_lmsk = (x_lmsk / chosen_line_wave - 1) * region.c
                axes[0,i].fill_between(v_lmsk, ymin, ymax, alpha=0.3, zorder=1, facecolor='orange')
                axes[1,i].fill_between(v_lmsk, ymin2, ymax2, alpha=0.3, zorder=1, facecolor='orange')
    except KeyError:
        print(f"\nno masked regions to plot for {line}.")

# define the current working directory
current_direc = os.getcwd()
# define the results directory based on the sub-folder name
results_dir = os.path.join(current_direc, f"replots/{region.folder_name}/")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
# define the filename and save it in the sub-directory output_files
filepath = os.path.join(results_dir, region.file_name + '_fittings.pdf')
fig.savefig(filepath, dpi=300, bbox_inches='tight')
plt.clf() # not showing the matplotlib figure; check the "plots" folder
############################### main section for replotting ###############################






