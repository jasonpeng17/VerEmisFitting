# README file

# Please refer to the `Versatile_Emission_Line_Fitting_Package_Guideline.pdf` in the `documents` folder for more details. Below are the key steps for installation and execution. 

## Installation

For installing the astroconda-lmfit environment:

1. With an environment definition YML file, users can build the environment with:
```
conda env create -f environment.yml
```

2. Activating an environment: 
```
conda activate astroconda-lmfit
```

3. To deactivate an environment:
```
conda deactivate
```

**Optional:**

- To delete an environment with all of its packages:
```
conda env remove -n astroconda-lmfit
```

- To rename a conda environment:
```
conda create --name astroconda-lmfit --clone astroconda2
```

## Running

For running the line-fitting algorithm:

In the `run_line_fitting.py` file, please prepare and input the flux, error, and wavelength 1D arrays. Users also need to input the redshift of the specific galaxy and determine whether the input spectrum is in "vac" (Vacuum) or "air" (Air) wavelength space.

**Example:**

```python
# Prepare and input the flux, error, and wavelength arrays
data_fits = current_direc + '/example_inputs/j1044+0353_addALL_icubes_wn.fits'
err_fits = current_direc + '/example_inputs/j1044+0353_addALL_icubes_wn_err.fits'
spec = fits.open(data_fits)[0].data
err = fits.open(err_fits)[0].data   
wave = np.load(current_direc + '/example_inputs/wave_grid.npy')

# Redshift of the galaxy
redshift = 0.01287

# Whether the spectrum is in "vacuum" or "air" wavelength space
vac_or_air = 'vac'

# whether the spectrum is in "vacuum" or "air" wavelength space
vac_or_air = 'vac'

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

input_example_txt = '/Users/zixuanpeng/Desktop/multiple_line_fitting_test/input_txt/line_selection_example.txt'
region = line_fitting_exec(redshift = redshift, vac_or_air = vac_or_air, E_BV = None, fits_name = data_fits, line_select_method = 'txt', input_txt = input_example_txt)
region.all_lines_result(wave, spec, err, n_iteration = 1000, get_flux = True, get_corr = False, save_flux_table = True, 
                        get_ew = True, save_ew_table = True, get_error = True, save_par_table = True)

```
Users can select the intended lines for fitting either by using the line-selection GUI (line_select_method = `gui`) or by inputting a text file (line_select_method = `txt`). Another GUI will appear for users to input the initial guess and range for each parameter.


**Saved Results:**

Checking the line-fitting outputs:

1. For the `fitting_plot()` function, if `savefig = True`, the result is saved as a .pdf file in the `plots` subfolder.
2. For the `all_lines_result()` function:
* If `save_flux_table = True`, the best-fitting line flux table is saved in the `flux_tables` subfolder.
* If `save_ew_table = True`, the best-fitting equivalent width table is saved in the `ew_tables` subfolder.
* If `save_par_table = True`, the best-fitting parameter table is saved in the `parameter_tables` subfolder.





