import numpy as np
from scipy.stats import f
import re

# line fitting model and residual functions 

def gaussian_1p_v(x, x0, sigma, a):
    """
    Calculate a single Gaussian profile in velocity space.

    Parameters:
    x (array-like): Input x data (in velocity space)
    x0 (float): Center of the Gaussian
    sigma (float): Width of the Gaussian
    a (float): Amplitude of the Gaussian

    Returns:
    model (array-like): Gaussian profile
    """
    # in velocity space
    model = a * np.exp(-(x - x0)**2 / (2. * sigma **2)) 
    return model

def lorentzian_1p_v(x, x0, sigma, a):
    '''
    This function calculates the value of a one-dimensional Lorentzian function given the input parameters x, x0, sigma, and a. 
    The function returns the model value of the Lorentzian.
    '''
    model = (a * ((sigma**2) / ((x - x0)**2 + sigma**2))) 
    return model

def gaussian_2p_v_doublet(params, x, x2, num_comp_first = 1, num_comp_second = 1):
    """
    Calculate a multi Gaussian model for the blended two peaks like [OII] 3726,29 in velocity space.

    Parameters:
    x (array-like): Input x data (in velocity space) for the first line
    x2 (array-like): Input x data (in velocity space) for the second line
    num_comp_first: the number of emission components for the first blended peak (1, 2, or 3).
    num_comp_second: the number of emission components for the second blended peak (1, 2, or 3).

    Returns:
    models (tuple): model for each blended peak and the model for the whole double peak line profiles.  
    """
    # in velocity space
    x0, sigma, a, a2 = params[:4]
    model_line1 = gaussian_1p_v(x, x0, sigma, a) 
    model_line2 = gaussian_1p_v(x2, x0, sigma, a2)
    model = model_line1 + model_line2

    # double component 
    if 2 in (num_comp_first, num_comp_second) and 3 not in (num_comp_first, num_comp_second):
        x0_1, sigma_1, a_1 = params[4:7]
        if num_comp_first == 2 and num_comp_second == 1:
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1)

        if num_comp_second == 2 and num_comp_first == 1:
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_1)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_1)
        
        if (num_comp_first == 2) and (num_comp_second == 2):
            a_2 = params[7]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) 
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_1, sigma_1, a_2)

    # triple component
    if 3 in (num_comp_first, num_comp_second):
        if num_comp_first == 3 and num_comp_second == 1:
            x0_1, sigma_1, a_1 = params[4:7]
            x0_2, sigma_2, a_2 = params[7:10]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x, x0_2, sigma_2, a_2)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x, x0_2, sigma_2, a_2)

        if num_comp_second == 3 and num_comp_first == 1:
            x0_1, sigma_1, a_1 = params[4:7]
            x0_2, sigma_2, a_2 = params[7:10]
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_2, sigma_2, a_2)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_2, sigma_2, a_2)

        if (num_comp_first == 3) and (num_comp_second == 2):
            x0_1, sigma_1, a_1, a_2 = params[4:8]
            x0_2, sigma_2, a_3 = params[8:11]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x, x0_2, sigma_2, a_3)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) 
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x, x0_2, sigma_2, a_3) + gaussian_1p_v(x2, x0_1, sigma_1, a_2) 

        if (num_comp_first == 2) and (num_comp_second == 3):
            x0_1, sigma_1, a_1, a_2 = params[4:8]
            x0_2, sigma_2, a_3 = params[8:11]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) 
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x2, x0_2, sigma_2, a_3)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x2, x0_2, sigma_2, a_3)

        if (num_comp_first == 3) and (num_comp_second == 3):
            x0_1, sigma_1, a_1, a_2 = params[4:8]
            x0_2, sigma_2, a_3, a_4 = params[8:12]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x, x0_2, sigma_2, a_3)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2)  + gaussian_1p_v(x2, x0_2, sigma_2, a_4)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x, x0_2, sigma_2, a_3) 
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_2)  + gaussian_1p_v(x2, x0_2, sigma_2, a_4)

    return model_line1, model_line2, model

def gaussian_2p_v(x, x0, x02, sigma, sigma2, a, a2):
    """
    Calculate a double-Gaussian profile in velocity space.

    Parameters:
    x (array-like): Input x data (in velocity space)
    x0 (float): Center of the first Gaussian
    x02 (float): Center of the second Gaussian
    sigma (float): Width of the first Gaussian
    sigma2 (float): Width of the second Gaussian
    a (float): Amplitude of the first Gaussian
    a2 (float): Amplitude of the second Gaussian

    Returns:
    model (array-like): doublet-Gaussian profile
    """
    # in velocity space
    model = gaussian_1p_v(x, x0, sigma, a) + gaussian_1p_v(x, x02, sigma2, a2)
    return model

def gaussian_3p_v(x, x0, x02, x03, sigma, sigma2, sigma3, a, a2, a3):
    """
    Calculate a triple-Gaussian profile in velocity space.

    Parameters:
    x (array-like): Input x data (in velocity space)
    x0 (float): Center of the first Gaussian
    x02 (float): Center of the second Gaussian
    x03 (float): Center of the third Gaussian
    sigma (float): Width of the first Gaussian
    sigma2 (float): Width of the second Gaussian
    sigma3 (float): Width of the third Gaussian
    a (float): Amplitude of the first Gaussian
    a2 (float): Amplitude of the second Gaussian
    a3 (float): Amplitude of the third Gaussian

    Returns:
    model (array-like): triple-Gaussian profile
    """
    # in velocity space
    model = gaussian_1p_v(x, x0, sigma, a) + gaussian_1p_v(x, x02, sigma2, a2) + gaussian_1p_v(x, x03, sigma3, a3)
    return model

def gaussian_3p_v_triplet(params, x, x2, x3, num_comp_first = 1, num_comp_second = 1, num_comp_third = 1):
    """
    Calculate a multi-Gaussian model for the blended three peaks like NII]&H&[NII] 6548&alpha&6583 in velocity space.

    Parameters:
    x (array-like): Input x data (in velocity space) for the first line
    x2 (array-like): Input x data (in velocity space) for the second line
    x3 (array-like): Input x data (in velocity space) for the third line
    num_comp_first: the number of emission components for the first blended peak (1, 2, or 3).
    num_comp_second: the number of emission components for the second blended peak (1, 2, or 3).
    num_comp_third: the number of emission components for the third blended peak (1, 2, or 3).

    Returns:
    models (tuple): model for each blended peak and the model for the whole triple peak line profiles.  
    """
    # in velocity space
    x0, sigma, a, a2, a3 = params[:5]
    model_line1 = gaussian_1p_v(x, x0, sigma, a)
    model_line2 = gaussian_1p_v(x2, x0, sigma, a2)
    model_line3 = gaussian_1p_v(x3, x0, sigma, a3)
    model = model_line1 + model_line2 + model_line3

    # double component
    if 2 in (num_comp_first, num_comp_second, num_comp_third) and 3 not in (num_comp_first, num_comp_second, num_comp_third):
        x0_1, sigma_1, a_1 = params[5:8]
        if num_comp_first == 2 and num_comp_second == 1 and num_comp_third == 1:
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1)

        if num_comp_first == 1 and num_comp_second == 2 and num_comp_third == 1:
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_1)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_1)

        if num_comp_first == 1 and num_comp_second == 1 and num_comp_third == 2:
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_1)
            model += gaussian_1p_v(x3, x0_1, sigma_1, a_1)

        if num_comp_first == 2 and num_comp_second == 2 and num_comp_third == 1:
            a_2 = params[8]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) 
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_1, sigma_1, a_2)

        if num_comp_first == 1 and num_comp_second == 2 and num_comp_third == 2:
            a_2 = params[8]
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_1)
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_2)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + gaussian_1p_v(x3, x0_1, sigma_1, a_2)

        if num_comp_first == 2 and num_comp_second == 1 and num_comp_third == 2:
            a_2 = params[8]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1)
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_2)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x3, x0_1, sigma_1, a_2)
        
        if num_comp_first == 2 and num_comp_second == 2 and num_comp_third == 2:
            a_2, a_3 = params[8:10]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2)
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_3)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x3, x0_1, sigma_1, a_3)

    # triple component    
    if 3 in (num_comp_first, num_comp_second, num_comp_third):
        if num_comp_first == 3 and num_comp_second == 1 and num_comp_third == 1:
            x0_1, sigma_1, a_1 = params[5:8]
            x0_2, sigma_2, a_2 = params[8:11]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x, x0_2, sigma_2, a_2)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x, x0_2, sigma_2, a_2)

        if num_comp_first == 1 and num_comp_second == 3 and num_comp_third == 1:
            x0_1, sigma_1, a_1 = params[5:8]
            x0_2, sigma_2, a_2 = params[8:11]
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_2, sigma_2, a_2)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_2, sigma_2, a_2)

        if num_comp_first == 1 and num_comp_second == 3 and num_comp_third == 2:
            x0_1, sigma_1, a_1, a_2 = params[5:9]
            x0_2, sigma_2, a_3 = params[9:12]
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_2)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_2, sigma_2, a_3)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + gaussian_1p_v(x3, x0_1, sigma_1, a_2) + gaussian_1p_v(x2, x0_2, sigma_2, a_3)

        if num_comp_first == 2 and num_comp_second == 3 and num_comp_third == 1:
            x0_1, sigma_1, a_1, a_2 = params[5:9]
            x0_2, sigma_2, a_3 = params[9:12]
            model_line1 += gaussian_1p_v(x1, x0_1, sigma_1, a_1)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x2, x0_2, sigma_2, a_3)
            model += gaussian_1p_v(x1, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x2, x0_2, sigma_2, a_3)

        if num_comp_first == 1 and num_comp_second == 1 and num_comp_third == 3:
            x0_1, sigma_1, a_1 = params[5:8]
            x0_2, sigma_2, a_2 = params[8:11]
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_1) + gaussian_1p_v(x3, x0_2, sigma_2, a_2)
            model += gaussian_1p_v(x3, x0_1, sigma_1, a_1) + gaussian_1p_v(x3, x0_2, sigma_2, a_2)

        if num_comp_first == 3 and num_comp_second == 2 and num_comp_third == 2:
            x0_1, sigma_1, a_1, a_2, a_3 = params[5:10]
            x0_2, sigma_2, a_4 = params[10:13]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x, x0_2, sigma_2, a_4)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) 
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_3)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x, x0_2, sigma_2, a_4)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x3, x0_1, sigma_1, a_3)

        if num_comp_first == 2 and num_comp_second == 3 and num_comp_third == 2:
            x0_1, sigma_1, a_1, a_2, a_3 = params[5:10]
            x0_2, sigma_2, a_4 = params[10:13]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) 
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x2, x0_2, sigma_2, a_4)
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_3)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x3, x0_1, sigma_1, a_3)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x2, x0_2, sigma_2, a_4)

        if num_comp_first == 2 and num_comp_second == 2 and num_comp_third == 3:
            x0_1, sigma_1, a_1, a_2, a_3 = params[5:10]
            x0_2, sigma_2, a_4 = params[10:13]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) 
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2)
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_3) + gaussian_1p_v(x3, x0_2, sigma_2, a_4)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_1, sigma_1, a_2)
            model += gaussian_1p_v(x3, x0_1, sigma_1, a_3) + gaussian_1p_v(x3, x0_2, sigma_2, a_4)
        
        if num_comp_first == 3 and num_comp_second == 3 and num_comp_third == 3:
            x0_1, sigma_1, a_1, a_2, a_3 = params[5:10]
            x0_2, sigma_2, a_4, a_5, a_6 = params[10:15]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x, x0_2, sigma_2, a_4)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x2, x0_2, sigma_2, a_5)
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_3) + gaussian_1p_v(x3, x0_2, sigma_2, a_6)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x3, x0_1, sigma_1, a_3)
            model += gaussian_1p_v(x, x0_2, sigma_2, a_4) + gaussian_1p_v(x2, x0_2, sigma_2, a_5) + gaussian_1p_v(x3, x0_2, sigma_2, a_6)

    return model_line1, model_line2, model_line3, model

def gaussian_lorentz_2p_v(x, x0, x02, sigma, sigma2, a, a2):
    """
    Gaussian and Lorentzian sum model.

    Args:
        x (array): x-data array.
        x0 (float): Gaussian center.
        x02 (float): Lorentzian center.
        sigma (float): Gaussian width.
        sigma2 (float): Lorentzian width.
        a (float): Gaussian amplitude.
        a2 (float): Lorentzian amplitude.

    Returns:
        model (array): Model evaluated at x-data points.
    """
    model = gaussian_1p_v(x, x0, sigma, a) + lorentzian_1p_v(x, x02, sigma2, a2)
    return model

def residual_1p_v_f(params, x, y, yerr):
    """
    Calculate the residuals for a single peak Gaussian model (fitting).

    Args:
        params (dict): Dictionary of fitting parameters.
        x (array): x-data array.
        y (array): y-data array.
        yerr (array): y-error array.

    Returns:
        residuals (array): Residuals array for the model.
    """
    # use for fitting
    x0 = params['center']
    sigma = params['sigma']
    a = params['amp']

    model = gaussian_1p_v(x, x0, sigma, a)
    return ((y-model)) / (yerr)

def residual_1p_v_c(params, x, y, yerr):
    """
    Calculate the residuals for a single peak Gaussian model (calculation).

    Args:
        params (list): List of parameters (x0, sigma, a).
        x (array): x-data array.
        y (array): y-data array.
        yerr (array): y-error array.

    Returns:
        residuals (array): Residuals array for the model.
    """
    # use for calculation
    x0,sigma,a = params

    model = gaussian_1p_v(x, x0, sigma, a)
    return ((y-model)) / (yerr)

def residual_2p_v_c(params, x, y, yerr):
    """
    Calculate the residuals for a multi Gaussian model (calculation) for a doublet like [OII] 3726,29.

    Args:
        params (list): List of parameters (x0, x02, sigma, sigma2, a, a2).
        x (array): x-data array.
        y (array): y-data array.
        yerr (array): y-error array.

    Returns:
        residuals (array): Residuals array for the model.
    """
    # use for calculation
    x0,x02,sigma,sigma2,a,a2 = params

    model = gaussian_2p_v(x, x0, x02, sigma, sigma2, a, a2)
    return ((y-model)) / (yerr)

def residual_2p_v_c_doublet(params, x, x2, y, yerr, num_comp_first = 1, num_comp_second = 1):
    """
    Calculate the residuals for a two peak Gaussian model for a doublet like [OII] 3726,29 (or other lines that have two peaks that needs to be fitted together).

    Args:
        params (list): List of parameters (x0, sigma, a, a2).
        x (array): x-data array for [OII] 3726.
        x2 (array): shifted x-data array for [OII] 3729.
        y (array): y-data array.
        yerr (array): y-error array.

    Returns:
        residuals (array): Residuals array for the model.
    """
    model = gaussian_2p_v_doublet(params, x, x2, num_comp_first=num_comp_first, num_comp_second=num_comp_second)[-1]        
    return (y - model) / yerr


def residual_2p_gl_v_c(params, x, y, yerr):
    """
    Calculate the residuals for a Gaussian-Lorentzian two peak model (calculation).

    Args:
        params (list): List of parameters (x0, x02, sigma, sigma2, a, a2).
        x (array): x-data array.
        y (array): y-data array.
        yerr (array): y-error array.

    Returns:
        residuals (array): Residuals array for the model.
    """
    # use for calculation
    x0,x02,sigma,sigma2,a,a2 = params

    model = gaussian_lorentz_2p_v(x, x0, x02, sigma, sigma2, a, a2)
    return ((y-model) / yerr)

def residual_3p_v_c(params, x, y, yerr):
    """
    Calculate the residuals for a three peak Gaussian model (calculation).

    Args:
        params (list): List of parameters (x0, x02, x03, sigma, sigma2, sigma3, a, a2, a3)
        x (array): x-data array.
        y (array): y-data array.
        yerr (array): y-error array.

    Returns:
        residuals (array): Residuals array for the model.

    """
    # use for calculation
    x0,x02,x03,sigma,sigma2,sigma3,a,a2,a3 = params

    model = gaussian_3p_v(x, x0, x02, x03, sigma, sigma2, sigma3, a, a2, a3)
    return ((y-model)) / (yerr)

def residual_3p_v_c_triplet(params, x, x2, x3, y, yerr, num_comp_first = 1, num_comp_second = 1, num_comp_third = 1):
    """
    Calculate the residuals for a multi Gaussian model for a triplet (or other lines that have three peaks that needs to be fitted together).

    Args:
        params (list): List of parameters (x0, sigma, a, a2, a3) and etc.
        x (array): x-data array for the first peak.
        x2 (array): shifted x-data array for the second peak.
        x3 (array): shifted x-data array for the third peak.
        y (array): y-data array.
        yerr (array): y-error array.

    Returns:
        residuals (array): Residuals array for the model.
    """
    model = gaussian_3p_v_triplet(params, x, x2, x3, num_comp_first = num_comp_first, num_comp_second = num_comp_second, num_comp_third = num_comp_third)[-1]        
    return (y - model) / yerr


# Function to split lines with multiple ions and wavelengths
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

# f-test function for testing whether a multiple-component fitting performs statistically better than a single-component fitting 
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

def extract_line_pars(filename):
    """
    Extract line parameters from a given file for the FitParamsWindow class.

    The function parses the file for specific line parameters such as selected_lines,
    fitting_method, multi_emis_lines, and so forth. Commented lines (starting with '#')
    in the file are skipped.

    Parameters:
    - filename (str): Name of the file containing the line parameters.

    Returns:
    - dict: Dictionary containing the extracted line parameters.

    Notes:
    - If a parameter in the file is assigned 'None', it's set as an empty list (for 'lines' 
      parameters) or as None (for others) in the returned dictionary.
    - If a parameter in the file is assigned 'True' or 'False', it's converted to the respective
      Boolean value in the returned dictionary.
    """
    parameters = {
        'selected_lines': None,
        'fitting_method': None,
        'multi_emis_lines': None,
        'double_gauss_lines': None,
        'triple_gauss_lines': None,
        'absorption_lines': None,
        'double_gauss_broad': None,
        'triple_gauss_broad': None
    }

    with open(filename, 'r') as file:
        for line in file:
            # Skip lines with comments (starting with '#')
            if line.strip().startswith("#"):
                continue

            for parameter in parameters.keys():
                if parameter in line:
                    if 'None' in line and 'lines' in parameter:
                        parameters[parameter] = []
                    elif 'None' in line:
                        parameters[parameter] = None
                    elif 'True' in line or 'False' in line:
                        parameters[parameter] = True if 'True' in line else False
                    elif parameter == 'fitting_method':
                        parameters[parameter] = re.search(r'fitting_method = "(.*?)"', line).group(1)
                    else:
                        parameters[parameter] = re.findall(r"'(.*?)'", line)

    return parameters


