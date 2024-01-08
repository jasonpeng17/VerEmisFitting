import numpy as np
from scipy.stats import f
from astropy import constants as const

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



