import numpy as np

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
    model = a * ((sigma**2) / ((x - x0)**2 + sigma**2)) 
    return model

def gaussian_lorentzian_2p_v_doublet(params, x, x2, num_comp_first = 1, num_comp_second = 1):
    """
    Calculate a Gaussian-Lorentzian model for the blended two peaks like [OII] 3726,29 in velocity space. The Lorentzian model is specifically used for fitting the ling wings in
    some lines (especially strong lines) that cannot be well-fitted by a Gaussian model. Therefore, only the "wing" part is fitted by the Lorentzian model; the Gaussian model is used 
    to fit the "core" part by default.

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
            model_line1 += lorentzian_1p_v(x, x0_1, sigma_1, a_1)
            model += lorentzian_1p_v(x, x0_1, sigma_1, a_1)

        if num_comp_second == 2 and num_comp_first == 1:
            model_line2 += lorentzian_1p_v(x2, x0_1, sigma_1, a_1)
            model += lorentzian_1p_v(x2, x0_1, sigma_1, a_1)
        
        if (num_comp_first == 2) and (num_comp_second == 2):
            a_2 = params[7]
            model_line1 += lorentzian_1p_v(x, x0_1, sigma_1, a_1) 
            model_line2 += lorentzian_1p_v(x2, x0_1, sigma_1, a_2)
            model += lorentzian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x2, x0_1, sigma_1, a_2)

    # triple component
    if 3 in (num_comp_first, num_comp_second):
        if num_comp_first == 3 and num_comp_second == 1:
            x0_1, sigma_1, a_1 = params[4:7]
            x0_2, sigma_2, a_2 = params[7:10]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x, x0_2, sigma_2, a_2)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x, x0_2, sigma_2, a_2)

        if num_comp_second == 3 and num_comp_first == 1:
            x0_1, sigma_1, a_1 = params[4:7]
            x0_2, sigma_2, a_2 = params[7:10]
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + lorentzian_1p_v(x2, x0_2, sigma_2, a_2)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + lorentzian_1p_v(x2, x0_2, sigma_2, a_2)

        if (num_comp_first == 3) and (num_comp_second == 2):
            x0_1, sigma_1, a_1, a_2 = params[4:8]
            x0_2, sigma_2, a_3 = params[8:11]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x, x0_2, sigma_2, a_3)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) 
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x, x0_2, sigma_2, a_3) + gaussian_1p_v(x2, x0_1, sigma_1, a_2) 

        if (num_comp_first == 2) and (num_comp_second == 3):
            x0_1, sigma_1, a_1, a_2 = params[4:8]
            x0_2, sigma_2, a_3 = params[8:11]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) 
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + lorentzian_1p_v(x2, x0_2, sigma_2, a_3)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_1, sigma_1, a_2) + lorentzian_1p_v(x2, x0_2, sigma_2, a_3)

        if (num_comp_first == 3) and (num_comp_second == 3):
            x0_1, sigma_1, a_1, a_2 = params[4:8]
            x0_2, sigma_2, a_3, a_4 = params[8:12]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x, x0_2, sigma_2, a_3)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2)  + lorentzian_1p_v(x2, x0_2, sigma_2, a_4)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x, x0_2, sigma_2, a_3) 
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_2)  + lorentzian_1p_v(x2, x0_2, sigma_2, a_4)

    return model_line1, model_line2, model

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

def gaussian_lorentzian_3p_v_triplet(params, x, x2, x3, num_comp_first = 1, num_comp_second = 1, num_comp_third = 1):
    """
    Calculate a multi-Gaussian model for the blended three peaks like [NII]&H&[NII] 6548&alpha&6583 in velocity space. The Lorentzian model is specifically used for fitting the ling wings in
    some lines (especially strong lines) that cannot be well-fitted by a Gaussian model. Therefore, only the "wing" part is fitted by the Lorentzian model; the Gaussian model is used 
    to fit the "core" part by default.

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
            model_line1 += lorentzian_1p_v(x, x0_1, sigma_1, a_1)
            model += lorentzian_1p_v(x, x0_1, sigma_1, a_1)

        if num_comp_first == 1 and num_comp_second == 2 and num_comp_third == 1:
            model_line2 += lorentzian_1p_v(x2, x0_1, sigma_1, a_1)
            model += lorentzian_1p_v(x2, x0_1, sigma_1, a_1)

        if num_comp_first == 1 and num_comp_second == 1 and num_comp_third == 2:
            model_line3 += lorentzian_1p_v(x3, x0_1, sigma_1, a_1)
            model += lorentzian_1p_v(x3, x0_1, sigma_1, a_1)

        if num_comp_first == 2 and num_comp_second == 2 and num_comp_third == 1:
            a_2 = params[8]
            model_line1 += lorentzian_1p_v(x, x0_1, sigma_1, a_1) 
            model_line2 += lorentzian_1p_v(x2, x0_1, sigma_1, a_2)
            model += lorentzian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x2, x0_1, sigma_1, a_2)

        if num_comp_first == 1 and num_comp_second == 2 and num_comp_third == 2:
            a_2 = params[8]
            model_line2 += lorentzian_1p_v(x2, x0_1, sigma_1, a_1)
            model_line3 += lorentzian_1p_v(x3, x0_1, sigma_1, a_2)
            model += lorentzian_1p_v(x2, x0_1, sigma_1, a_1) + lorentzian_1p_v(x3, x0_1, sigma_1, a_2)

        if num_comp_first == 2 and num_comp_second == 1 and num_comp_third == 2:
            a_2 = params[8]
            model_line1 += lorentzian_1p_v(x, x0_1, sigma_1, a_1)
            model_line3 += lorentzian_1p_v(x3, x0_1, sigma_1, a_2)
            model += lorentzian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x3, x0_1, sigma_1, a_2)
        
        if num_comp_first == 2 and num_comp_second == 2 and num_comp_third == 2:
            a_2, a_3 = params[8:10]
            model_line1 += lorentzian_1p_v(x, x0_1, sigma_1, a_1)
            model_line2 += lorentzian_1p_v(x2, x0_1, sigma_1, a_2)
            model_line3 += lorentzian_1p_v(x3, x0_1, sigma_1, a_3)
            model += lorentzian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x2, x0_1, sigma_1, a_2) + lorentzian_1p_v(x3, x0_1, sigma_1, a_3)

    # triple component    
    if 3 in (num_comp_first, num_comp_second, num_comp_third):
        if num_comp_first == 3 and num_comp_second == 1 and num_comp_third == 1:
            x0_1, sigma_1, a_1 = params[5:8]
            x0_2, sigma_2, a_2 = params[8:11]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x, x0_2, sigma_2, a_2)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x, x0_2, sigma_2, a_2)

        if num_comp_first == 1 and num_comp_second == 3 and num_comp_third == 1:
            x0_1, sigma_1, a_1 = params[5:8]
            x0_2, sigma_2, a_2 = params[8:11]
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + lorentzian_1p_v(x2, x0_2, sigma_2, a_2)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + lorentzian_1p_v(x2, x0_2, sigma_2, a_2)

        if num_comp_first == 1 and num_comp_second == 3 and num_comp_third == 2:
            x0_1, sigma_1, a_1, a_2 = params[5:9]
            x0_2, sigma_2, a_3 = params[9:12]
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_2)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + lorentzian_1p_v(x2, x0_2, sigma_2, a_3)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_1) + gaussian_1p_v(x3, x0_1, sigma_1, a_2) + lorentzian_1p_v(x2, x0_2, sigma_2, a_3)

        if num_comp_first == 2 and num_comp_second == 3 and num_comp_third == 1:
            x0_1, sigma_1, a_1, a_2 = params[5:9]
            x0_2, sigma_2, a_3 = params[9:12]
            model_line1 += gaussian_1p_v(x1, x0_1, sigma_1, a_1)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + lorentzian_1p_v(x2, x0_2, sigma_2, a_3)
            model += gaussian_1p_v(x1, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_1, sigma_1, a_2) + lorentzian_1p_v(x2, x0_2, sigma_2, a_3)

        if num_comp_first == 1 and num_comp_second == 1 and num_comp_third == 3:
            x0_1, sigma_1, a_1 = params[5:8]
            x0_2, sigma_2, a_2 = params[8:11]
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_1) + lorentzian_1p_v(x3, x0_2, sigma_2, a_2)
            model += gaussian_1p_v(x3, x0_1, sigma_1, a_1) + lorentzian_1p_v(x3, x0_2, sigma_2, a_2)

        if num_comp_first == 3 and num_comp_second == 2 and num_comp_third == 2:
            x0_1, sigma_1, a_1, a_2, a_3 = params[5:10]
            x0_2, sigma_2, a_4 = params[10:13]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x, x0_2, sigma_2, a_4)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) 
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_3)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x, x0_2, sigma_2, a_4)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x3, x0_1, sigma_1, a_3)

        if num_comp_first == 2 and num_comp_second == 3 and num_comp_third == 2:
            x0_1, sigma_1, a_1, a_2, a_3 = params[5:10]
            x0_2, sigma_2, a_4 = params[10:13]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) 
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + lorentzian_1p_v(x2, x0_2, sigma_2, a_4)
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_3)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x3, x0_1, sigma_1, a_3)
            model += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + lorentzian_1p_v(x2, x0_2, sigma_2, a_4)

        if num_comp_first == 2 and num_comp_second == 2 and num_comp_third == 3:
            x0_1, sigma_1, a_1, a_2, a_3 = params[5:10]
            x0_2, sigma_2, a_4 = params[10:13]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) 
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2)
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_3) + lorentzian_1p_v(x3, x0_2, sigma_2, a_4)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_1, sigma_1, a_2)
            model += gaussian_1p_v(x3, x0_1, sigma_1, a_3) + lorentzian_1p_v(x3, x0_2, sigma_2, a_4)
        
        if num_comp_first == 3 and num_comp_second == 3 and num_comp_third == 3:
            x0_1, sigma_1, a_1, a_2, a_3 = params[5:10]
            x0_2, sigma_2, a_4, a_5, a_6 = params[10:15]
            model_line1 += gaussian_1p_v(x, x0_1, sigma_1, a_1) + lorentzian_1p_v(x, x0_2, sigma_2, a_4)
            model_line2 += gaussian_1p_v(x2, x0_1, sigma_1, a_2) + lorentzian_1p_v(x2, x0_2, sigma_2, a_5)
            model_line3 += gaussian_1p_v(x3, x0_1, sigma_1, a_3) + lorentzian_1p_v(x3, x0_2, sigma_2, a_6)
            model += gaussian_1p_v(x, x0_1, sigma_1, a_1) + gaussian_1p_v(x2, x0_1, sigma_1, a_2) + gaussian_1p_v(x3, x0_1, sigma_1, a_3)
            model += lorentzian_1p_v(x, x0_2, sigma_2, a_4) + lorentzian_1p_v(x2, x0_2, sigma_2, a_5) + lorentzian_1p_v(x3, x0_2, sigma_2, a_6)

    return model_line1, model_line2, model_line3, model

def gaussian_3p_v_triplet(params, x, x2, x3, num_comp_first = 1, num_comp_second = 1, num_comp_third = 1):
    """
    Calculate a multi-Gaussian model for the blended three peaks like [NII]&H&[NII] 6548&alpha&6583 in velocity space.

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
    Gaussian and Lorentzian sum model (2 peaks).

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

def gaussian_lorentz_3p_v(x, x0, x02, x03, sigma, sigma2, sigma3, a, a2, a3):
    """
    Gaussian and Lorentzian sum model (3 peaks). The "core" is fitted by 2 Gaussian models, while the "wings" part is fitted by 1 Lorentzian model.

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
    model = gaussian_1p_v(x, x0, sigma, a) + gaussian_1p_v(x, x02, sigma2, a2) +lorentzian_1p_v(x, x03, sigma3, a3)
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

def residual_2p_gl_v_c_doublet(params, x, x2, y, yerr, num_comp_first = 1, num_comp_second = 1):
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
    model = gaussian_lorentzian_2p_v_doublet(params, x, x2, num_comp_first=num_comp_first, num_comp_second=num_comp_second)[-1]        
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

def residual_3p_gl_v_c(params, x, y, yerr):
    """
    Calculate the residuals for a three-peak Gaussian-Lorentzian model (calculation).

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

    model = gaussian_lorentz_3p_v(x, x0, x02, x03, sigma, sigma2, sigma3, a, a2, a3)
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

def residual_3p_gl_v_c_triplet(params, x, x2, x3, y, yerr, num_comp_first = 1, num_comp_second = 1, num_comp_third = 1):
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
    model = gaussian_lorentzian_3p_v_triplet(params, x, x2, x3, num_comp_first = num_comp_first, num_comp_second = num_comp_second, num_comp_third = num_comp_third)[-1]        
    return (y - model) / yerr