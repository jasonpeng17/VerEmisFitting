import numpy as np
from scipy.stats import f
from astropy import constants as const
from astropy.stats import sigma_clip, sigma_clipped_stats

def region_around_line(w, flux, cont, order = 1):
    '''
    Fit the local continuum level with a n-order polynomial around the chosen line profile within the selected local continuum regions.
    '''
    #index is true in the region where we fit the polynomial
    indcont = ((w >= cont[0][0]) & (w <= cont[0][1])) |((w >= cont[1][0]) & (w <= cont[1][1]))
    #index of the region we want to return
    indrange = (w >= cont[0][0]) & (w <= cont[1][1])
    # make a flux array of shape
    # (number of spectra, number of points in indrange)
    f = np.zeros(indrange.sum())

    # use sigma-clip to get rid of sharp features (outliers)
    filtered_flux = sigma_clip(flux[indcont], sigma=3, maxiters=None, cenfunc='median', masked=True, copy=False)
    filtered_wave = np.ma.compressed(np.ma.masked_array(w[indcont], filtered_flux.mask))
    filtered_flux_data = filtered_flux.compressed()
    # fit polynomial of second order to the continuum region
    linecoeff, covar = np.polyfit(filtered_wave, filtered_flux_data, order, full = False, cov = True)
    flux_cont = np.polyval(linecoeff, w[indrange])

    # estimate the error of flux_cont
    # calculate the residual between the continuum and the polynomial, and then calculate the standard deviation of the residual
    flux_cont_out = np.polyval(linecoeff, filtered_wave)
    cont_resid = filtered_flux_data - flux_cont_out
    flux_cont_err = np.abs(np.nanstd(cont_resid)) * np.ones(len(flux_cont))

    # divide the flux by the polynomial and put the result in our
    # new flux array
    f[:] = flux[indrange]/np.polyval(linecoeff, w[indrange])
    return w[indrange], f, flux_cont, flux_cont_err, linecoeff, covar

def extract_key_parts_from_ratio(ratio):
    """Helper function to extract the amplitude names from the dictionary that contains the fixed amplitude ratios"""
    parts = ratio.split('_over_')
    
    # Extract parts from the provided ratio string, preserving any suffixes.
    first_part = parts[0].split('_')[1:]
    second_part = parts[1].split('_')[0:]

    return ['_'.join(first_part), '_'.join(second_part)]

def num_fix_amps(line_name, amps_fixed_names):
    """Helper function to determine the number of fixed amplitude parameters for a specific line."""
    if '&' in line_name: # multiplet that should be fitted together
        multilet_lines = split_multilet_line(line_name)
        multilet_lines_no_ion = [line.split(' ')[1] for line in multilet_lines] # get rid of the ion name prefix
        n_fix = sum([line in fixed_name for line in multilet_lines_no_ion for fixed_name in amps_fixed_names])
    else: # singlet 
        single_line_no_ion = line_name.split(' ')[1] # get rid of the ion name prefix
        n_fix = sum([single_line_no_ion in fixed_name for fixed_name in amps_fixed_names])
    return n_fix

def num_fix_vels(line_name, fitting_method, vel_fixed_num, broad_wings_lines, triple_gauss_lines, absorption_lines):
    """Helper function to determine the number of fixed velocity parameters for a specific line."""
    n_fix = 0
    fix_num = vel_fixed_num[fitting_method]
    
    if '&' in line_name:  # Handling multiplet lines
        multilet_lines = split_multilet_line(line_name)
        # Check if any of the lines in the multiplet satisfies the conditions
        n_fix += fix_num if any(line in absorption_lines for line in multilet_lines) else 0
        n_fix += fix_num if any(line in broad_wings_lines for line in multilet_lines) else 0
        n_fix += fix_num if any(line in triple_gauss_lines for line in multilet_lines) else 0
    else:  # Handling single lines
        n_fix += fix_num if line_name in absorption_lines else 0
        n_fix += fix_num if line_name in broad_wings_lines else 0
        n_fix += fix_num if line_name in triple_gauss_lines else 0
    return n_fix

def lorentz_integral_analy(a, sigma):
    '''
    Return the analytical integral of 1D Lorentzian profile. 
    '''
    bracket_term = 1j * (np.log(-1j / sigma) - np.log(1j / sigma))
    return a * sigma * bracket_term

def lorentz_integral_analy_err(a, sigma, a_err, sigma_err):
    '''
    Return the analytical integral of 1D Lorentzian profile. 
    '''
    bracket_term = 1j * (np.log(-1j / sigma) - np.log(1j / sigma))
    term1 = (sigma * bracket_term * a_err)**2
    term2 = (a * bracket_term * sigma_err)**2
    return np.sqrt(term1 + term2)

def split_multilet_line(line):
    '''
    Splits a line containing multiple ions and wavelengths into individual components.

    Parameters:
    line (str): The line to split. Ions and wavelengths should be separated by a space, 
                and multiple ions or wavelengths should be separated by an '&'.

    Returns:
    list: A list of strings, each containing an ion and a wavelength.
    '''
    ion_parts, wave_parts = line.split(' ')
    ions = ion_parts.split('&')
    waves = wave_parts.split('&')
    lines = [f"{ion} {wave}" for ion, wave in zip(ions, waves)]
    return lines

def f_test(chi1, chi2, n, p1=3, p2=6, alpha =0.003):
    '''
    Performs an F-test to determine whether a multiple-component fitting statistically 
    outperforms a single-component fitting.

    Parameters:
    chi1, chi2 (float): The chi-square values of the single-component and multiple-component models, respectively.
    n (int): The number of data points.
    p1, p2 (int, optional): The number of parameters in the single-component and multiple-component models, respectively.
    alpha (float, optional): The significance level for the F-test.

    Returns:
    bool: True if the multiple-component model statistically outperforms the single-component model, False otherwise.
    '''
    dof1 = n - p1
    dof2 = n - p2
    f_theo = f.ppf(q=1-alpha, dfn=dof1-dof2, dfd=dof2, loc=0, scale=1)
    f_ober = ((chi1 - chi2)/(dof1 - dof2)) / (chi2/dof2)
    print(f_theo, f_ober)
    # print(f_theo, f_ober)
    if f_ober > f_theo:
        # two or three gaussian model is better
        return True
    else:
        # two or three gaussian model doesn't provide a statistically better result
        return False

def calc_ew(model_w, wave_line, cont_line):
    '''
    Calculates the equivalent width (EW) of a spectral line.

    Parameters:
    model_w (array_like): The model spectrum in wavelength space (background subtracted).
    wave_line (array_like): The wavelengths of the spectral line profile.
    cont_line (float): The background continuum level.

    Returns:
    float: The equivalent width of the spectral line.
    '''
    flux_all = (-model_w) / cont_line
    ew_all = np.abs(flux_all[:-1] * np.diff(wave_line))
    ew_all = ew_all.sum()
    return ew_all

def calc_ew_err(model_w, sigma_w, cont_line):
    '''
    Calculates the error of the equivalent width (EW) of a spectral line.

    Parameters:
    model_w (array_like): The model spectrum in wavelength space (background subtracted).
    sigma_w (array_like): The standard deviation of the model spectrum.
    cont_line (float): The continuum level.

    Returns:
    float: The error of the equivalent width of the spectral line.
    '''
    FWHM_all = 2 * sigma_w * np.sqrt(2 * np.log(2))
    r_i_all = (-model_w) / cont_line
    epsilon_all = np.sqrt(np.sum((r_i_all - np.mean(r_i_all))**2) / len(r_i_all))
    ew_all_err = 1.5 * np.sqrt(FWHM_all * 0.5) * epsilon_all
    return ew_all_err

def calc_com(v_arr, flux_arr):
    """
    Calculate the center of mass of a line profile in velocity space.

    Parameters:
    - v_arr (array-like): Array of velocities.
    - flux_arr (array-like): Array of fluxes corresponding to the velocities.

    Returns:
    - float: Center of mass of the line profile.

    Notes:
    - This function uses numpy's `nansum` to handle any NaN values.
    """
    com = np.nansum(v_arr * flux_arr) / np.nansum(flux_arr) # avoid any nan values
    return com

def find_two_peaks(lst):
    """
    Find the two highest values in a given list.

    Parameters:
    - lst (list): A list of numerical values.

    Returns:
    - list/str: If the list has at least two elements, returns a list containing the two highest values. 
                If the list has fewer than two elements, returns a string indicating that at least two elements are required.

    Notes:
    - This function sorts the input list in descending order and selects the first two elements.
    """
    if len(lst) < 2:
        return "List must have at least two elements."
    
    sorted_lst = sorted(lst, reverse=True)
    return sorted_lst[:2]

def delta_v_to_delta_lam(delta_v, lam_0):
    """
    Convert a velocity difference (delta_v) in km/s to a wavelength difference (delta_lam) in angstroms.

    Parameters:
    - delta_v (float): Velocity difference in kilometers per second.
    - lam_0 (float): Reference wavelength in angstroms.

    Returns:
    - float: Corresponding wavelength difference in angstroms.

    Notes:
    - Utilizes the light speed constant from the astropy.constants package.
    """
    delta_lam = (delta_v / const.c.to('km/s').value) * lam_0
    return delta_lam

def closest_indx_arr(arr, value):
    """
    Find the index of the element in an array closest to a specified value.

    Parameters:
    - arr (array-like): An array of numerical values.
    - value (float): The value to find the closest element to.

    Returns:
    - int: Index of the element in `arr` closest to `value`.

    Notes:
    - The function converts the input to a numpy array for processing.
    """
    arr = np.asarray(arr)
    index = (np.abs(arr - value)).argmin()
    return index

def window_size_in_lam(delta_v, lam_0, scale_factor = 2., return_indx = False, wave_arr = None):
    """
    Determine the default window size for fitting in wavelength space.

    Parameters:
    - delta_v (float): Velocity difference in km/s.
    - lam_0 (float): Reference wavelength in angstroms.
    - scale_factor (float, optional): Scaling factor for the window size. Defaults to 2.0.
    - return_indx (bool, optional): Whether to return indices instead of wavelength values. Defaults to False.
    - wave_arr (array-like, optional): Array of wavelength values. Required if return_indx is True.

    Returns:
    - list: If return_indx is False, returns [lower wavelength limit, upper wavelength limit].
            If return_indx is True, returns [index of lower wavelength limit, index of upper wavelength limit].

    Notes:
    - The function asserts that both `return_indx` and `wave_arr` must be either both set or both unset.
    - Utilizes `delta_v_to_delta_lam` for conversion from velocity to wavelength.
    """
    # Assert that both return_indx and wave_arr should have the same boolean value
    assert (return_indx == (wave_arr is not None)), "Both 'return_indx' and 'wave_arr' should either be truthy or falsy."

    delta_lam = delta_v_to_delta_lam(delta_v, lam_0)
    lam_lolim = lam_0 - delta_lam * scale_factor # lower wavelength limit
    lam_uplim = lam_0 + delta_lam * scale_factor # upper wavelength limit

    if return_indx and (wave_arr is not None): # return the indices corresponding to the lower and upper wave limits
        indx_lolim = closest_indx_arr(wave_arr, lam_lolim)
        indx_uplim = closest_indx_arr(wave_arr, lam_uplim)
        return [indx_lolim, indx_uplim]
    else: # return the lower and upper wave limits
        return [lam_lolim, lam_uplim]

def local_cont_reg(wave_arr, indx_lolim, indx_uplim, fraction = 0.1):
    """
    Determine the default left and right local continuum regions for fitting.

    Parameters:
    - wave_arr (array-like): Array of wavelength values.
    - indx_lolim (int): Lower index limit for the fitting window.
    - indx_uplim (int): Upper index limit for the fitting window.
    - fraction (float, optional): Fraction of the window to consider for the continuum. Defaults to 0.1.

    Returns:
    - list: A list containing two lists: [[left continuum start, left continuum end], [right continuum start, right continuum end]].

    Notes:
    - The function selects continuum regions based on the provided fraction of the wavelength fitting window.
    """
    wave_fit_window = wave_arr[indx_lolim:indx_uplim+1]
    x1, x2 = wave_fit_window[0], wave_fit_window[-1]
    x3, x4 = np.nanpercentile(wave_fit_window, fraction * 100), np.nanpercentile(wave_fit_window, 100 - fraction * 100)
    return [[x1, x3],[x4, x2]]

def aic_lmfit(chisqr, nvarys, ndata):
    """Definition of AIC from LMFIT: https://github.com/astrodee/threadcount/
       ndata * np.log(Chisqr/ndata) + 2 * nvarys."""
    return ndata * np.log(chisqr/ndata) + 2 * nvarys

def bic_lmfit(chisqr, nvarys, ndata):
    """Definition of BIC from LMFIT: https://github.com/astrodee/threadcount/
       ndata * np.log(Chisqr/ndata) + np.log(ndata) * nvarys."""
    return ndata * np.log(chisqr/ndata) + np.log(ndata) * nvarys

def aic_duvet(chisqr, nvarys):
    """Definition of AIC from the DUVET survey: https://github.com/astrodee/threadcount/
       Chisqr + 2 * nvarys."""
    return chisqr + 2 * nvarys

def bic_duvet(chisqr, nvarys, ndata):
    """Definition of BIC from the DUVET survey: https://github.com/astrodee/threadcount/
       Chisqr + np.log(ndata) * nvarys."""
    return chisqr + np.log(ndata) * nvarys



