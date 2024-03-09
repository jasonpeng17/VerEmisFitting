# README file

### Please refer to the `Versatile_Emission_Line_Fitting_Package_Guideline.pdf` in the `doc` folder for more details. Below are the key steps for installation and execution. 

## Installation

For installing the ver_emis_fitting environment:

1. With an environment definition YML file, users can build the environment with:
```
conda env create -f environment.yml
```

2. Activating an environment: 
```
conda activate ver_emis_fitting
```

3. To deactivate an environment:
```
conda deactivate
```

**Optional:**

- To delete an environment with all of its packages:
```
conda env remove -n ver_emis_fitting
```

- To rename a conda environment:
```
conda create --name <new environment name> --clone ver_emis_fitting
```

## Running

For running the line-fitting algorithm:

In the `run_line_fitting.py` file, please prepare the flux, error, and wavelength 1D arrays. Users also need to input the redshift of the specific galaxy and determine whether the input spectrum is in "vac" (Vacuum) or "air" (Air) wavelength space.

**Example:**

```python
# Prepare the flux, error, and wavelength arrays
data_fits = current_direc + '/example_inputs/j1044+0353_addALL_icubes_wn.fits'
err_fits = current_direc + '/example_inputs/j1044+0353_addALL_icubes_wn_err.fits'
spec = fits.open(data_fits)[0].data
err = fits.open(err_fits)[0].data   
wave = np.load(current_direc + '/example_inputs/wave_grid.npy')

# run the fitting result for each selected line profile
# redshift of the galaxy
redshift = 0.01287
# whether the spectrum is in "vac" or "air" wavelength space
vac_or_air = 'air'
# the order of fitting local continuum level
fit_cont_order = 1
# "line_selection_method = 'gui' or 'txt' defines how you want to select the lines for fitting, whether using a GUI or inputting a txt"
line_select_method = 'txt'
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
file_name = "test" # if None, then file_name = folder_name; else, file_name will be f"{folder_name}_{file_name}"
region = line_fitting_exec(redshift = redshift, vac_or_air = vac_or_air, folder_name = folder_name, file_name = file_name, line_select_method = line_select_method, 
                           input_txt = input_example_txt, fit_cont_order = fit_cont_order, fit_window_gui = fit_window_gui, params_windows_gui = params_windows_gui)

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

```
Users can select the intended lines for fitting either by using the line-selection GUI (`line_select_method = gui`) or by inputting a text file (`line_select_method = txt`). Another GUI will appear for users to input the initial guess and range for each parameter.


**Saved Results:**

Checking the line-fitting outputs:

1. For the `fitting_plot()` function, if `savefig = True`, the result is saved as a .pdf file in the `plots` subfolder.
2. For the `all_lines_result()` function:
* If `save_flux_table = True`, the best-fitting line flux table is saved in the `flux_tables` subfolder.
* If `save_ew_table = True`, the best-fitting equivalent width table is saved in the `ew_tables` subfolder.
* If `save_par_table = True`, the best-fitting parameter table is saved in the `parameter_tables` subfolder.
* If `save_stats_table = True`, the best-fitting statistics table is saved in the `stats_tables` subfolder.
* If `save_cont_params_table = True`, the best-fitting parameter table for continuum fit is saved in the `cont_tables` subfolder.

## Include More Emission Lines / Modify Default Line-Fitting Window

The `vac_wavelengths.txt` or `air_wavelengths.txt` file in the `doc` folder contain the lines available for fitting, as shown in the line-selection GUI. Users can modify these files by adding new lines or editing existing ones, following the provided format. For example, `[OIII] 4363: 4364.44` indicates that the line `[OIII] 4363` has a (vacuum) wavelength of `4364.44` Angstroms. It's recommended to categorize each line under the specific element section (e.g., `<Oxygen>` for `[OIII] 4363`), so that the added line appears under that element in the line-selection GUI. Comments can be included by starting a line with '#'.

After adding a particular line to both wavelength files, users should also add this line to the `default_window_vspace.txt` file in the `doc` folder such that the pipeline can generate a default line-fitting window for this line. For example, `[OIII] 4363: 600` means the line `[OIII] 4363` has a half velocity width of the line-fitting window of 800 km/s (or equivalently, a total velocity width of 1600 km/s). **The left and right local continuum regions are, by default, set to the first and last 10% of the wavelength range of the line-fitting window, respectively.**

## Recommended Fitting Procedures

0. **Check Available Emission Lines**: Refer to the `vac_wavelengths.txt` or `air_wavelengths.txt` file in the `doc` folder to verify if the desired emission lines are available. If not, follow the steps in the **Include More Emission Lines / Modify Default Line-Fitting Window** section to add new lines. Remember to also add these new lines to the `default_window_vspace.txt` file in the `doc` folder to generate a default line-fitting window.

1. **Prepare Data and Set Parameters**: As demonstrated in the **Running** section, prepare the flux, error, and wavelength arrays. Adjust the input parameters accordingly. Choose the method to select lines for fitting (`line_select_method = gui` or `txt`) and specify the number of Gaussian functions for each line. Refer to the example in the `input_txt` folder and the documentation in the `doc` folder for guidance.

2. **(Optional) Fix the Amplitude Ratio Between two lines**: To ensure physical fitting results for two lines, it is advisable to fix the amplitude ratio between them, consistent with the theoretical line flux ratio. For example, `[OIII] 5007 / [OIII] 4959 = 2.98`. This adjustment can be implemented through the `params_windows`GUI or directly within the `input_txt` file. Adopting this strategy ensures that the line flux ratio remains constant, given that these lines are tied together in the velocity space (same velocity widths) .

3. **Interactive Fitting Window Adjustment**: It is recommended to initially set `fit_window_gui = True` in the `line_fitting_exec` class. This will launch a `bokeh` server, allowing interactive selection of the fitting window and masking of regions with sharp features. The fitting window and local continuum region information will be saved to the `cont_dir` folder as a `.cont` file, specifying four boundary points in Angstroms: `x1` (lower-end of the fitting window), `x2` (upper-end of the fitting window), `x3` (upper-end of the left local continuum region), and `x4` (lower-end of the right local continuum region). These define the local continuum regions as **x1 - x3** and **x4 - x2**. Modify the `fit_cont_order` parameter in the `line_fitting_exec` class to set the polynomial order for fitting the local continuum. Masked region information will be stored in the `lmsk_dir` folder as a `.lmsk` file, with each line defining a region to be masked during fitting. Masked regions will also be shaded as **orange** in the best-fitting plot.

4. **Review Saved Results**: Refer to the **Saved Results** section under **Running** to examine the fitting outcomes, including the PDF file in the `plots` folder and other relevant tables such as flux tables in the `flux_tables`, equivalent width tables in the `ew_tables`, and best-fitting parameter tables in the `parameter_tables`.

5. **Refitting with Saved Settings**: For refitting the same selected lines, the saved `.cont` and `.lmsk` files will be loaded to recreate the previously chosen fitting window, local continuum regions, and masked regions. Therefore, you can set `fit_window_gui = False` in the `line_fitting_exec` class. 

6. **(Optional) Potential Application to IFU Data**: Importing the VerEmisFitting package for fitting spectra extracted from IFU data can be time-consuming and redundant, as it requires repeatedly clicking buttons in every pop-up GUI for each spectrum. Therefore, it is recommended to set `line_select_method = 'txt'`, `fit_window_gui = False`, and `params_window_gui = False`. The first two parameters are illustrated in the previous steps. The last parameter (`params_window_gui`) controls whether to interactively determine the initial values and ranges of parameters for each iteration. With this setup, no interactive GUI will pop up; thus, it is essential to ensure that the default configuration adequately fits your data. For each extracted spectrum from the IFU data cube, it is recommended to assign a unique folder name, `folder_name`, or file name, `file_name`, within the `line_fitting_exec` class to ensure the saved products are uniquely named. If the local continuum regions or the masked regions differ significantly across these extracted spectra, assigning a unique folder name for each spectrum is recommended. This approach is preferred because the program is designed to find unique `cont_dir` and `lmsk_dir` folders with each respective spectrum.





