import numpy as np
import re

def merge_element_waves(elements_dict):
    '''
    Merge all wavelength dictionaries from different elements into a single dictionary.
    '''
    merged_wave_dict = {}
    for element in elements_dict:
        merged_wave_dict.update(elements_dict[element])
    return merged_wave_dict

def read_wavelengths_from_file(filename, redshift):
    '''
    Read wavelengths and ion names from air or vacuum wavelength files; 
    Return a dictionary of elements, each containing a wavelength dictionary 
    with the ion key names and the wavelength values (in observed frame).
    '''
    elements_dict = {}
    current_element = None

    element_pattern = re.compile(r"<(.+)>")
    wavelength_pattern = re.compile(r"([^:]+):\s*(.+)")

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            element_match = element_pattern.match(line)
            if element_match:
                current_element = element_match.group(1)
                elements_dict[current_element] = {}
            else:
                wavelength_match = wavelength_pattern.match(line)
                if wavelength_match and current_element is not None:
                    key = wavelength_match.group(1).strip()
                    waves = [float(wave) for wave in wavelength_match.group(2).split()]
                    waves_0 = [wave * (1 + redshift) for wave in waves]
                    elements_dict[current_element][key] = np.array(waves_0)

    return elements_dict

def read_window_width_from_file(filename):
    '''
    Read half velocity width of fitting window for each line; 
    Return the dictionary with the line names (key) and the half velocity width (values).
    '''
    v_width_half_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            if not line.strip() or line.strip().startswith("#"):
                continue
            parts = line.split(':')
            if len(parts) < 2:  # Check if the line has at least two parts
                continue  # Skip lines that don't follow the expected format
            key = parts[0].strip()
            try:
                v_width_half = float(parts[1].strip())  # Convert the second part to float
                v_width_half_dict[key] = v_width_half
            except ValueError:
                # Handle lines where the second part is not a valid float
                print(f"Warning: Unable to parse '{line.strip()}' as a valid entry.")
    return v_width_half_dict

def extract_amps_ratio(amp_num_list, amp_den_list, ratio_list, double_gauss_list, triple_gauss_list):
    """
    Extract sets of lines that have fixed amplitude ratios from amp_num_list, amp_den_list, and ratio_list.
    """
    # initialize the returned dictionary
    amps_ratio_dict = dict()
    for amp_num, amp_den, ratio in zip(amp_num_list, amp_den_list, ratio_list):
        if ((amp_num in double_gauss_list) and (amp_den in double_gauss_list)) or \
           ((amp_num in double_gauss_list) and (amp_den in triple_gauss_list)) or \
           ((amp_num in triple_gauss_list) and (amp_den in double_gauss_list)): # two velocity components
            line_comps = ["", "_b"]
        elif (amp_num in triple_gauss_list) and (amp_den in triple_gauss_list): # three velocity components
            line_comps = ["", "_b", "_b2"]
        else: # single velocity component
            line_comps = [""]
        for line_comp in line_comps:
            amp_num_comp = amp_num.split(" ")[1] + line_comp
            amp_den_comp = amp_den.split(" ")[1]+ line_comp
            amps_ratio_dict[f"ratio_{amp_num_comp}_over_{amp_den_comp}"] = ratio
    return amps_ratio_dict

def extract_line_pars(filename, return_amps_fixed_dict = False):
    """
    Extract line parameters from a given file for the FitParamsWindow class.

    The function parses the file for specific line parameters such as selected_lines,
    fitting_method, multi_emis_lines, and so forth. Commented lines (starting with '#')
    in the file are skipped.

    Parameters:
    - filename (str): Name of the file containing the line parameters.
    - return_amps_fixed_dict (bool): Return the dictionary that contains the lines with fixed amplitude ratios.

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
        'triple_gauss_broad': None,
        'lorentz_bw_em_lines': None,
        'lorentz_bw_abs_lines': None,
        'amp_num_lines' : None,
        'amp_den_lines' : None,
        'amp_fixed_ratio': None
    }

    with open(filename, 'r') as file:
        for line in file:
            # Skip lines with comments (starting with '#')
            if line.strip().startswith("#"):
                continue

            for parameter in parameters.keys():
                if parameter in line:
                    if 'None' in line and ('lines' in parameter or 'ratio' in parameter):
                        parameters[parameter] = []
                    elif 'None' in line:
                        parameters[parameter] = None
                    elif 'True' in line or 'False' in line:
                        parameters[parameter] = True if 'True' in line else False
                    elif parameter == 'fitting_method':
                        parameters[parameter] = re.search(r'fitting_method = "(.*?)"', line).group(1)
                    elif parameter == "amp_fixed_ratio":
                        parameters[parameter] = [float(ratio) for ratio in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
                    else:
                        parameters[parameter] = re.findall(r"'(.*?)'", line)

    # return the fixed amplitude ratio dictionary
    if return_amps_fixed_dict:
        double_gauss_list, triple_gauss_list = parameters['double_gauss_lines'], parameters['triple_gauss_lines']
        amp_num_list, amp_den_list, ratio_list = parameters['amp_num_lines'], parameters['amp_den_lines'], parameters['amp_fixed_ratio']
        amps_ratio_dict = extract_amps_ratio(amp_num_list, amp_den_list, ratio_list, double_gauss_list, triple_gauss_list)
        return parameters, amps_ratio_dict
    else:
        return parameters
