### GUI for users to input the initial guess and range value for each parameter

import tkinter as tk
# from tkinter import ttk
import ttkbootstrap as ttk
import tkinter.font as font
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
from matplotlib.figure import Figure
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.visualization import LinearStretch, LogStretch, SinhStretch, SqrtStretch
from astropy.visualization import ZScaleInterval, MinMaxInterval
from astropy.visualization import ImageNormalize
from matplotlib.widgets import Button, TextBox
from tkinter.simpledialog import askinteger
from tkinter.simpledialog import askstring
from tkinter.filedialog import askopenfilename

from IPython import embed

class FitParamsWindow(tk.Tk):
    """
    A window for defining initial guess parameters and their ranges for various spectral components.

    This class presents an interface allowing the user to define initial guess values for
    several spectral components, like center, sigma, and amplitude. Components considered
    include a first emission component, a second emission emission, a third emission component,
    and an absorption component.

    The window consists of multiple text entry fields arranged according to the provided
    spectral lines or spectral features. It will also produce a secondary window allowing
    the user to define the range around the initial guess that will be considered in fitting.

    Note: The units for the inputs of 'center_{}' and 'sigma_{}' are in km/s. The inputs
    for all amplitudes, 'amp_{}', are in units of the maximum of the line flux array, i.e.,
    a value of 0.5 means 0.5 * max(flux_v_arr).

    Key Attributes:
    -----------
        selected_lines (list): List of selected emission lines for the first component.
        broad_wings_lines (list): List of emission lines for the multi emission component.
        triple_gauss_lines (list): List of emission lines for the triple Gaussian fit.
        absorption_lines (list): List of absorption lines.

        params_dict (dict): Dictionary holding the tk Entry widgets for the guess values.
        params_range_dict (dict): Dictionary holding the tk Entry widgets for the parameter ranges.
        
        params_values (dict): Dictionary holding the saved guess values after user input.
        params_range_values (dict): Dictionary holding the saved range values after user input.

        amplitude_ratios(dict): Dictionary holding the ratio values for all amplitude pairs after user input.

    Methods:
    -----------
        create_entry: Creates a label and entry widget for a specific parameter.
        create_amp_entry: Creates entry widgets for amplitude parameters of given spectral lines.
        create_section_label: Creates a section label for different spectral components.
        create_range_window: Produces a new window for defining parameter ranges.
        create_ratio_window: Produces a new window for defining the ratio between two amplitudes. 
        add_combo_row: Adds a new row in the create_ratio_window to define the ratio for another pair of amplitudes.
        delete_combo_row: Deletes the latest created row in case the row is mistakenly added.
        save_ratios: Stores the ratio values for all amplitude pairs entered by the user.
        save_params: Stores the values of the guess parameters entered by the user.
        save_ranges: Stores the range values of the parameters entered by the user.
        stop_mainloop: Stops the tkinter mainloop.
        run: Initiates the tkinter mainloop and returns the saved parameter dictionaries.

    Example:
        window = FitParamsWindow(selected_lines_list, broad_wings_list, triple_gauss_list, absorption_list)
        guess_params, range_params = window.run()
    """
    def __init__(self, selected_lines, broad_wings_lines, triple_gauss_lines, absorption_lines, params_windows_gui=True):
        super().__init__()

        self.withdraw() #delete the tk.Tk blank window

        self.params_windows_gui = params_windows_gui # pop up the params window or not; default is True

        self.selected_lines = selected_lines
        self.broad_wings_lines = broad_wings_lines
        self.triple_gauss_lines = triple_gauss_lines
        self.absorption_lines = absorption_lines

        self.params_dict = {}
        self.params_range_dict = {}

        if self.params_windows_gui:
            # font style
            self.font_tuple = ("Helvetica", 14, "bold")
            self.font_txt_tuple = ("Helvetica", 16, "italic")

            self.guess_window = ttk.Toplevel(self)
            self.guess_window.title("Input Initial Guess for each Parameter")

            # Create a Label widget for the message
            self.guess_message = ttk.Label(self.guess_window, text="Notice: the units for the inputs of 'center_{}' and 'sigma_{}' are in km/s. \
                                                                   The inputs for all amplitudes, 'amp_{}' are in units of the maximum of the line flux array, max(flux_v_arr), e.g., 0.5 means 0.5 * max(flux_v_arr).",
                                          font=self.font_txt_tuple, wraplength=500, justify=tk.LEFT)
            self.guess_message.grid(row=0, column=0, sticky='w')

            self.guess_column = 0
            self.guess_row = 0

            self.range_window = None

            # Create a new frame to contain all the widgets that were directly in guess_window
            content_frame = ttk.Frame(self.guess_window)
            content_frame.grid(row=1, column=0, sticky='nsew')

            # Emission component
            if self.selected_lines:
                self.create_section_label('First Emission Component', content_frame)
                self.create_entry('center_e', 0, content_frame)
                self.create_entry('sigma_e', 50, content_frame)
                for line in self.selected_lines:
                    self.create_amp_entry(line, '', 'e', content_frame)
                self.guess_column += 1
                self.guess_row = 1

            # Broad wings emission component
            if self.broad_wings_lines:
                self.create_section_label('Second Emission Component', content_frame)
                self.create_entry('center_b', 0, content_frame)
                self.create_entry('sigma_b', 150, content_frame)
                for line in self.broad_wings_lines:
                    self.create_amp_entry(line, '_b', 'b', content_frame)
                self.guess_column += 1
                self.guess_row = 1

            # Triple Gaussian fitting
            if self.triple_gauss_lines:
                self.create_section_label('Third Emission Component', content_frame)
                self.create_entry('center_b2', 0, content_frame)
                self.create_entry('sigma_b2', 400, content_frame)
                for line in self.triple_gauss_lines:
                    self.create_amp_entry(line, '_b2', 'b2', content_frame)
                self.guess_column += 1
                self.guess_row = 1

            # Absorption component
            if self.absorption_lines:
                self.create_section_label('Absorption Component', content_frame)
                self.create_entry('center_a', 0, content_frame)
                self.create_entry('sigma_a', 1200, content_frame)
                for line in self.absorption_lines:
                    self.create_amp_entry(line, '_abs', 'abs', content_frame)
                self.guess_column += 1
                self.guess_row = 1

            # button for fixing the ratio between two amplitudes
            ratio_button_frame = ttk.Frame(self.guess_window)
            ratio_button_frame.grid(row=2, column=0, sticky='ew')
            ttk.Button(ratio_button_frame, text="Set Amplitude Ratio", command=self.create_ratio_window, bootstyle= 'default').pack(side='bottom')

            # button for saving the enter input        
            button_frame = ttk.Frame(self.guess_window)
            button_frame.grid(row=3, column=0, sticky='ew')
            ttk.Button(button_frame, text="Save", command=self.save_params, bootstyle= 'success').pack()

            self.guess_window.protocol("WM_DELETE_WINDOW", self.stop_mainloop)
        else:
            self.set_default_values()

    def set_default_values(self):
        # initialize params' values and ranges
        self.params_values = {}
        self.params_range_values = {}

        # Emission component
        if self.selected_lines:
            # first emission component's center and sigma values and ranges
            self.params_values['center_e'] = 0
            self.params_values['sigma_e'] = 50
            self.params_range_values['center_e'] = 10
            self.params_range_values['sigma_e'] = 50

        # Broad wings emission component
        if self.broad_wings_lines:
            # second emission component's center and sigma values and ranges
            self.params_values['center_b'] = 0
            self.params_values['sigma_b'] = 150
            self.params_range_values['center_b'] = 10
            self.params_range_values['sigma_b'] = 100

        # Triple Gaussian fitting
        if self.triple_gauss_lines:
            # third emission component's center and sigma values and ranges
            self.params_values['center_b2'] = 0
            self.params_values['sigma_b2'] = 400
            self.params_range_values['center_b2'] = 10
            self.params_range_values['sigma_b2'] = 300

        # Absorption component
        if self.absorption_lines:
            # absorption component's center and sigma values and ranges
            self.params_values['center_a'] = 0
            self.params_values['sigma_a'] = 1200
            self.params_range_values['center_a'] = 10
            self.params_range_values['sigma_a'] = 500

        # for each selected line
        for line in self.selected_lines:
            if '&' not in line:
                amp = 'amp_{}'.format(line.split(' ')[1])
                self.params_values[amp] = 1.
                self.params_range_values[amp] = 1.
            else:
                for amp in line.split(' ')[1].split('&'):
                    amp_label = 'amp_{}'.format(amp) 
                    self.params_values[amp_label] = 1.
                    self.params_range_values[amp_label] = 1.
        # for each broad_wing line
        for broad_line in self.broad_wings_lines:
            amp_broad = 'amp_{}'.format(broad_line.split(' ')[1]) + '_b'
            self.params_values[amp_broad] = 0.05
            self.params_range_values[amp_broad] = 0.05
        # for each triple_gauss line
        for triple_line in self.triple_gauss_lines:
            amp_triple = 'amp_{}'.format(triple_line.split(' ')[1]) + '_b2'
            self.params_values[amp_triple] = 0.01
            self.params_range_values[amp_triple] = 0.01
        # for each abs_gauss line
        for abs_line in self.absorption_lines:
            amp_abs = 'amp_{}'.format(abs_line.split(' ')[1]) + '_abs'
            self.params_values[amp_abs] = -0.05
            self.params_range_values[amp_abs] = 0.05

        # Set default amplitude ratios if needed
        self.amplitude_ratios = {}
        # Example: self.amplitude_ratios['ratio_amp1_over_amp2'] = 1.0

    def create_entry(self, param_name, init_value, master, window='guess'):
        if window == 'guess':
            column = self.guess_column
            row = self.guess_row
        if window == 'range':
            column = self.range_column
            row = self.range_row

        tk.Label(master, text=param_name).grid(column=column, row=row, sticky='w')
        e = tk.Entry(master)
        e.grid(column=column, row=row + 1, sticky='w')
        e.insert(0, init_value)
        self.params_dict[param_name] = e
        if window == 'guess':
            self.guess_row += 2
        if window == 'range':
            self.range_row += 2

    def create_amp_entry(self, line, suffix, label_suffix, master, window='guess'):
        if '&' not in line:
            amp = 'amp_{}'.format(line.split(' ')[1]) + suffix
            if label_suffix == 'e':
                if window == 'guess':
                    amp_value = 1
                if window == 'range':
                    amp_value = 0.5
            if label_suffix == 'b':
                amp_value = 0.05
            if label_suffix == 'b2':
                amp_value = 0.01
            if label_suffix == 'abs':
                if window == 'guess':
                    amp_value = -0.1
                if window == 'range':
                    amp_value = 0.05
            self.create_entry(amp, amp_value, master, window)
        else:
            for amp in line.split(' ')[1].split('&'):
                amp_label = 'amp_{}'.format(amp) + suffix
                self.create_entry(amp_label, 1, master, window)

    def create_section_label(self, name, master, window='guess'):
        if window == 'guess':
            column = self.guess_column
            row = 0
        if window == 'range':
            column = self.range_column
            row = 0

        tk.Label(master, text=name, font=self.font_tuple).grid(column=column, row=row)
        if window == 'guess':
            self.guess_row = 1
        if window == 'range':
            self.range_row = 1

    def create_range_window(self):
        self.range_window = ttk.Toplevel(self)
        self.range_window.title("Input Range Size for each Parameter")

        # Create a Label widget for the message
        self.range_message = ttk.Label(self.range_window, text="Notice: the units for the inputs of 'center_{}' and 'sigma_{}' are in km/s. \
                                                               The inputs for all amplitudes, 'amp_{}' are in units of the maximum of the line flux array, max(flux_v_arr), e.g., 0.5 means 0.5 * max(flux_v_arr).", 
                                                               font=self.font_txt_tuple, wraplength=500, justify=tk.LEFT)
        self.range_message.grid(row=0, column=0, sticky='w')

        self.range_column = 0
        self.range_row = 0

        content_frame = ttk.Frame(self.range_window)
        content_frame.grid(row=1, column=0, sticky='nsew')

        if self.selected_lines:
            self.create_section_label('First Emission Component', content_frame, window='range')
            self.create_entry('center_e', 10, content_frame, window='range')
            self.create_entry('sigma_e', 50, content_frame, window='range')
            for line in self.selected_lines:
                self.create_amp_entry(line, '', 'e', content_frame, window='range')
            self.range_column += 1
            self.range_row = 1

        if self.broad_wings_lines:
            self.create_section_label('Second Emission Component', content_frame, window='range')
            self.create_entry('center_b', 10, content_frame, window='range')
            self.create_entry('sigma_b', 100, content_frame, window='range')
            for line in self.broad_wings_lines:
                self.create_amp_entry(line, '_b', 'b', content_frame, window='range')
            self.range_column += 1
            self.range_row = 1

        if self.triple_gauss_lines:
            self.create_section_label('Third Emission Component', content_frame, window='range')
            self.create_entry('center_b2', 10, content_frame, window='range')
            self.create_entry('sigma_b2', 300, content_frame, window='range')
            for line in self.triple_gauss_lines:
                self.create_amp_entry(line, '_b2', 'b2', content_frame, window='range')
            self.range_column += 1
            self.range_row = 1

        if self.absorption_lines:
            self.create_section_label('Absorption Component', content_frame, window='range')
            self.create_entry('center_a', 10, content_frame, window='range')
            self.create_entry('sigma_a', 500, content_frame, window='range')
            for line in self.absorption_lines:
                self.create_amp_entry(line, '_abs', 'abs', content_frame, window='range')
            self.range_column += 1
            self.range_row = 1

        button_frame = ttk.Frame(self.range_window)
        button_frame.grid(row=2, column=0, sticky='ew')
        
        ttk.Button(button_frame, text="Save", command=self.save_ranges, bootstyle = 'success').pack(side='bottom')

        self.range_window.protocol("WM_DELETE_WINDOW", self.stop_mainloop)

    def create_ratio_window(self):
        self.ratio_window = ttk.Toplevel(self)
        self.ratio_window.title("Set Amplitude Ratios")

        self.amplitude_options = list(self.params_dict.keys())
        self.amplitude_options = [amp for amp in self.amplitude_options if 'amp_' in amp]

        self.amplitude_combos_container = ttk.Frame(self.ratio_window)
        self.amplitude_combos_container.grid(row=0, column=0, columnspan=4)

        # List to track created rows
        self.rows = []

        # Initialize with one row of comboboxes
        self.add_combo_row()

        self.add_button = ttk.Button(self.ratio_window, text="Add", command=self.add_combo_row, bootstyle = 'primary')
        self.add_button.grid(row=1, column=0, columnspan=2)

        # Delete button to remove the last added row
        self.delete_button = ttk.Button(self.ratio_window, text="Delete", command=self.delete_combo_row, bootstyle = 'primary')
        self.delete_button.grid(row=1, column=2, columnspan=2)

        self.save_button = ttk.Button(self.ratio_window, text="Save Ratios", command=self.save_ratios, bootstyle = 'success')
        self.save_button.grid(row=2, column=0, columnspan=2)

        # Quit button to close the window without saving
        self.quit_button = ttk.Button(self.ratio_window, text="Quit Without Saving", command=self.quit_ratio_window, bootstyle = 'danger')
        self.quit_button.grid(row=2, column=2, columnspan=2)

        self.ratio_window.protocol("WM_DELETE_WINDOW", self.stop_mainloop)

    def add_combo_row(self):
        """Add a row in the self.create_ratio_window."""
        row_frame = ttk.Frame(self.amplitude_combos_container)
        
        # Append this frame to the rows list
        self.rows.append(row_frame)

        # Drop-down menu for the first amplitude:
        amp1_combo = ttk.Combobox(row_frame, values=self.amplitude_options, style='info.TCombobox', foreground='black')
        amp1_combo.grid(row=0, column=0)

        # Label for the divide sign
        tk.Label(row_frame, text="/").grid(row=0, column=1)

        # Drop-down menu for the second amplitude:
        amp2_combo = ttk.Combobox(row_frame, values=self.amplitude_options, style='info.TCombobox', foreground='black')
        amp2_combo.grid(row=0, column=2)

        # Label for the equals sign
        ttk.Label(row_frame, text="=").grid(row=0, column=3)

        # Entry field for the ratio:
        ratio_entry = ttk.Entry(row_frame)
        ratio_entry.grid(row=0, column=4)

        row_frame.pack(pady=5)

    def delete_combo_row(self):
        """Delete a row in the self.create_ratio_window."""
        if self.rows:
            last_row = self.rows.pop()
            last_row.destroy()

    def quit_ratio_window(self):
        """Close the ratio window without saving any ratios."""
        self.ratio_window.destroy()

    def extract_key_part(self, amp):
        parts = amp.split('_')
        key_part = parts[1]
        
        # Check if there's a "_b" or "_bNUMBER" suffix.
        if len(parts) > 2:
            key_part += '_' + '_'.join(parts[2:])
        
        return key_part

    def save_ratios(self):
        # if not hasattr(self, 'amplitude_ratios'):
        self.amplitude_ratios = {}

        for row_frame in self.amplitude_combos_container.winfo_children():
            # Assuming the comboboxes and entry are added in order in add_combo_row
            amp1_combo = row_frame.winfo_children()[0]
            amp2_combo = row_frame.winfo_children()[2]
            ratio_entry = row_frame.winfo_children()[4]

            amp1 = amp1_combo.get()
            amp2 = amp2_combo.get()

            try:
                ratio = float(ratio_entry.get())
                amp1_key_part = self.extract_key_part(amp1)
                amp2_key_part = self.extract_key_part(amp2)
                self.amplitude_ratios[f"ratio_{amp1_key_part}_over_{amp2_key_part}"] = ratio
                # self.amplitude_ratios[f"ratio_{amp1.split('_')[1]}_over_{amp2.split('_')[1]}"] = ratio
            except ValueError:
                tk.messagebox.showerror("Error", "Please enter a valid number for the ratio.")
                return  # Exit function without destroying window if error encountered

        self.ratio_window.destroy()

    def save_params(self):
        self.params_values = {k: float(v.get()) for k, v in self.params_dict.items()}  
        self.guess_window.destroy()  # Destroy the guess_window after saving
        self.create_range_window()

    def save_ranges(self):
        self.params_range_values = {k: float(v.get()) for k, v in self.params_dict.items()}  
        self.range_window.destroy()  # Destroy the range_window after saving
        self.quit()

    def stop_mainloop(self):
        self.quit()

    def run(self):
        if self.params_windows_gui:
            self.mainloop()
            return self.params_values, self.params_range_values, getattr(self, 'amplitude_ratios', {})
        else:
            return self.params_values, self.params_range_values, self.amplitude_ratios
            


