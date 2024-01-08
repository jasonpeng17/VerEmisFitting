### GUI for users to select lines to be fitted 

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
from analysis_utils import *
from params_gui import *

class LineSelector(tk.Tk):
    """
    A GUI application built on Tkinter for selecting emission lines and fitting methods.

    Attributes:
    -----------
    emission_lines : list
        List of emission lines to be displayed in the GUI.
    check_vars : list
        List of tkinter StringVar() objects corresponding to the emission lines.
    fitting_methods : list
        List of possible fitting methods.
    selected_method : tkinter.StringVar
        The fitting method selected by the user.
    label : tkinter.Label
        The label displayed at the top of the window.
    option_menu : tkinter.OptionMenu
        Dropdown menu for selecting fitting methods.
    button : tkinter.Button
        Button to finalize the line selection.
    
    Methods:
    --------
    update_selected_lines():
        Updates the list of selected emission lines based on the checkboxes ticked.
    select_broad_wings():
        Creates a new window for the user to select emission lines with multiple velocity components.
    select_double_gaussians():
        Creates a new window for the user to select lines that require double Gaussian fitting.
    select_triple_gaussians():
        Creates a new window for the user to select lines that require triple Gaussian fitting.
    select_absorption():
        Creates a new window for the user to select lines that have absorption troughs.
    get_selected_lines():
        Updates and finalizes the selected lines based on the user's input.
    run():
        Starts the tkinter main loop and returns the user's selections after the main loop has finished.

    Notes:
    ------
    The class also has various other internal attributes used for managing the GUI state, 
    such as the various Tkinter widgets and the lists of selected lines in different categories.
    """
    def __init__(self, elements_dict):
        super().__init__()

        # Font for all labels, buttons, etc.
        self.font_tuple = ("Helvetica", 14, "bold")  
        self.font_txt_tuple = ("Helvetica", 16, "italic")

        self.elements_dict = elements_dict
        self.check_vars = {}
        self.fitting_methods = ["Free fitting", "Fix velocity centroid", "Fix velocity width", "Fix velocity centroid and width"]
        self.selected_method = tk.StringVar()
        self.selected_method.set(self.fitting_methods[0]) 

        # Add a label at the top of the window
        self.label = ttk.Label(self, text="Select lines for fitting", font=self.font_tuple)
        self.label.grid(row=0, column=0, columnspan=5)

        # Iterate over elements and create sections
        row = 1
        for element, lines in self.elements_dict.items():
            # Create a section label for each element
            section_label = ttk.Label(self, text=element, font=self.font_tuple)
            section_label.grid(row=row, column=0, columnspan=5, sticky='w')
            row += 1

            # Create checkboxes for each line in the element
            for i, line in enumerate(lines):
                var = tk.StringVar(value="")
                self.check_vars[line] = var
                check = ttk.Checkbutton(self, text=line, variable=var, onvalue=line, offvalue="", command=self.update_selected_lines, bootstyle= 'primary')
                # Distribute checkboxes across five columns
                check.grid(row=row + i // 5, column=i % 5, sticky='w')

            row += (len(lines) - 1) // 5 + 2  # Update row index for next element

        # Create OptionMenu for fitting options
        self.option_menu = ttk.OptionMenu(self, self.selected_method, *self.fitting_methods, bootstyle= 'default')
        self.option_menu.grid(row=row, column=0, columnspan=5)
        row += 1

        # Add button to finalize selection
        self.button = ttk.Button(self, text="Confirm", command=self.select_broad_wings, bootstyle= 'success')
        self.button.grid(row=row, column=0, columnspan=5)

    def update_selected_lines(self):
        self.selected_lines = [line for line, var in self.check_vars.items() if var.get()]

    def select_broad_wings(self):
        self.update_selected_lines()
        self.selected_fitting_method = self.selected_method.get()

        # create a new window to select lines for fitting "broad wings"
        self.broad_wings_window = ttk.Toplevel(self)
        self.broad_wings_window.title("Select lines that have multiple emission velocity components")
        self.check_vars_broad_wings = []
        self.checks_broad_wings = []

        # define the row number
        row_i = 0

        for i, line in enumerate(self.selected_lines):
            # single line profile
            if '&' not in line:
                var = tk.StringVar(value="")
                self.check_vars_broad_wings.append(var)

                check = ttk.Checkbutton(self.broad_wings_window, variable=var, onvalue=line, offvalue="", bootstyle= 'primary')
                check.grid(row=row_i, column=0, sticky='w')
                self.checks_broad_wings.append(check)

                label = ttk.Label(self.broad_wings_window, text=line)
                label.grid(row=row_i, column=1, sticky='w')

                # increase the row number by 1
                row_i += 1
            
            # multi-component line profile that needs to be fitted together
            else:
                multilet_lines = split_multilet_line(line)
                for indx, line in enumerate(multilet_lines):
                    # create a check and a label for line1
                    var = tk.StringVar(value="")
                    self.check_vars_broad_wings.append(var)

                    check = ttk.Checkbutton(self.broad_wings_window, variable=var, onvalue=line, offvalue="", bootstyle= 'primary')
                    check.grid(row=row_i, column=0, sticky='w')
                    self.checks_broad_wings.append(check)

                    label = ttk.Label(self.broad_wings_window, text=line)
                    label.grid(row=row_i, column=1, sticky='w')

                    # increase the row number by 1
                    row_i += 1

        ttk.Button(self.broad_wings_window, text="Confirm", command=self.select_double_gaussians, bootstyle= 'success').grid(row=len(self.checks_broad_wings) + 1, column=0, columnspan=10)

        # Raise the window to the top
        self.broad_wings_window.lift()
        self.broad_wings_window.focus_force()
        self.broad_wings_window.minsize(500, 300)  # Set minimum size of the window

    def select_double_gaussians(self):
        self.broad_wings_lines = [var.get() for var in self.check_vars_broad_wings if var.get()]
        self.broad_wings_window.destroy()

        if not self.broad_wings_lines:
            # Skip directly to selecting absorption lines if no broad wings are selected for double Gaussian fitting
            self.select_fitting_function()
            return

        # Create a new window to select lines for fitting "double Gaussian"
        self.double_gauss_window = ttk.Toplevel(self)
        self.double_gauss_window.title("Select lines that require 'two-component' fitting")
        self.check_vars_double_gauss = [tk.StringVar(value="") for line in self.broad_wings_lines]

        for i in range(len(self.broad_wings_lines)):
            check = ttk.Checkbutton(self.double_gauss_window, variable=self.check_vars_double_gauss[i],
                                   onvalue=self.broad_wings_lines[i], offvalue="", bootstyle= 'primary')
            check.grid(row=i, column=0, sticky='w')
            label = ttk.Label(self.double_gauss_window, text=self.broad_wings_lines[i])
            label.grid(row=i, column=1, sticky='w')

        ttk.Button(self.double_gauss_window, text="Confirm", command=self.select_triple_gaussians, bootstyle= 'success').grid(row=i+1, column=0, columnspan=10)

        # Raise the window to the top
        self.double_gauss_window.lift()
        self.double_gauss_window.focus_force()
        self.double_gauss_window.minsize(500, 300)  # Set minimum size of the window

    def select_triple_gaussians(self):
        self.double_gauss_lines = [var.get() for var in self.check_vars_double_gauss if var.get()]
        self.double_gauss_window.destroy()

        if ((len(self.broad_wings_lines) == len(self.double_gauss_lines)) and (len(self.double_gauss_lines) >= 1) and (len(self.broad_wings_lines) >= 1)):
            # Skip directly to selecting fitting functions if only one broad wings line is selected and is selected as the "two-component" one
            self.select_fitting_function()
            return

        # Create a new window to select lines for fitting "triple Gaussian"
        self.triple_gauss_window = ttk.Toplevel(self)
        self.triple_gauss_window.title("Select lines that require 'three-component' fitting")
        self.check_vars_triple_gauss = [tk.StringVar(value="") for line in self.broad_wings_lines]

        for i in range(len(self.broad_wings_lines)):
            check = ttk.Checkbutton(self.triple_gauss_window, variable=self.check_vars_triple_gauss[i],
                                   onvalue=self.broad_wings_lines[i], offvalue="", bootstyle= 'primary')
            check.grid(row=i, column=0, sticky='w')
            label = tk.Label(self.triple_gauss_window, text=self.broad_wings_lines[i])
            label.grid(row=i, column=1, sticky='w')

        ttk.Button(self.triple_gauss_window, text="Confirm", command=self.select_fitting_function, bootstyle= 'success').grid(row=i+1, column=0, columnspan=10)

        # Raise the window to the top
        self.triple_gauss_window.lift()
        self.triple_gauss_window.focus_force()
        self.triple_gauss_window.minsize(500, 300)  # Set minimum size of the window

    def select_fitting_window(self, lines_to_fit, double_gauss_lines, triple_gauss_lines):
        # Create a new window to select the fitting function for broad wings
        self.fitting_function_window = ttk.Toplevel(self)
        self.fitting_function_window.title("Select the fitting function for broad wings")

        # Create a Label widget for the message
        self.bw_message = ttk.Label(self.fitting_function_window, 
                                   text="Notice: the 'core' part is fitted by Gaussian model(s) by default. The Lorentzian model is specifically used for fitting the ling wings in some lines (especially strong lines) that cannot be well-fitted by a Gaussian model.", 
                                   font=self.font_txt_tuple, wraplength=500, justify=tk.LEFT)
        self.bw_message.grid(row=0, column=0, sticky='w')

        self.fitting_functions_vars = {}
        for i, line in enumerate(lines_to_fit):
            # Determine the component type for the label
            component_type = "Two Velocity Components" if line in double_gauss_lines else "Three Velocity Components"
            self.fitting_functions_vars[line] = tk.StringVar(value="Gaussian")
            ttk.Label(self.fitting_function_window, text=f"{line} ({component_type}):").grid(row=i+1, column=0, sticky='w', padx=(10, 0))
            ttk.Radiobutton(self.fitting_function_window, text="Gaussian", variable=self.fitting_functions_vars[line], value="Gaussian").grid(row=i+1, column=1, sticky='w')
            ttk.Radiobutton(self.fitting_function_window, text="Lorentzian", variable=self.fitting_functions_vars[line], value="Lorentzian").grid(row=i+1, column=2, sticky='w')

        ttk.Button(self.fitting_function_window, text="Confirm", command=self.select_absorption, bootstyle= 'success').grid(row=len(lines_to_fit) + 1, column=0, columnspan=3)

        # Raise the window to the top
        self.fitting_function_window.lift()
        self.fitting_function_window.focus_force()
        self.fitting_function_window.minsize(300, 100 + len(lines_to_fit)*30)  # Set minimum size of the window

    def select_fitting_function(self):
        lines_to_fit = []

        if not hasattr(self, 'broad_wings_lines'):
            self.broad_wings_lines = []

        # Check for double Gaussian broad wings
        if hasattr(self, 'check_vars_double_gauss'):
            self.double_gauss_lines = [var.get() for var in self.check_vars_double_gauss if var.get()]
            self.double_gauss_window.destroy()
            if self.double_gauss_lines:
                self.double_gauss_broad = messagebox.askquestion("Confirmation", "Are there any broad wings in the lines that require 'two-component' fitting?")
                if self.double_gauss_broad == 'yes':
                    lines_to_fit.extend(self.double_gauss_lines)
            else:
                self.double_gauss_lines = []
                self.double_gauss_broad = 'no'
        else:
            self.double_gauss_lines = []
            self.double_gauss_broad = 'no'

        # Check for triple Gaussian broad wings
        if hasattr(self, 'check_vars_triple_gauss'):
            self.triple_gauss_lines = [var.get() for var in self.check_vars_triple_gauss if var.get()]
            self.triple_gauss_window.destroy()
            if self.triple_gauss_lines:
                self.triple_gauss_broad = messagebox.askquestion("Confirmation", "Are there any broad wings in the lines that require 'three-component' fitting?")
                if self.triple_gauss_broad == 'yes':
                    lines_to_fit.extend(self.triple_gauss_lines)
            else:
                self.triple_gauss_lines = []
                self.triple_gauss_broad = 'no'
        else:
            self.triple_gauss_lines = []
            self.triple_gauss_broad = 'no'

        # If there are lines to fit, call the fitting window function
        if lines_to_fit:
            self.select_fitting_window(lines_to_fit, self.double_gauss_lines, self.triple_gauss_lines)
        else:
            self.select_absorption()

    def select_absorption(self):
        try:
            self.fitting_functions_choices = {line: var.get() for line, var in self.fitting_functions_vars.items()}
            self.fitting_function_window.destroy()
        except:
            self.fitting_functions_choices = {}
        # create a new window to select lines for fitting "absorption"
        self.absorption_window = ttk.Toplevel(self)
        self.absorption_window.title("Select lines that have 'absorption troughs'")
        self.check_vars_absorption = [tk.StringVar(value="") for line in self.selected_lines]

        for i in range(len(self.selected_lines)):
            check = ttk.Checkbutton(self.absorption_window, variable=self.check_vars_absorption[i],
                                   onvalue=self.selected_lines[i], offvalue="", bootstyle= 'primary')
            check.grid(row=i, column=0, sticky='w')
            label = ttk.Label(self.absorption_window, text=self.selected_lines[i])
            label.grid(row=i, column=1, sticky='w')

        ttk.Button(self.absorption_window, text="Confirm", command=self.get_selected_lines, bootstyle= 'success').grid(row=i+1, column=0, columnspan=10)

        # Raise the window to the top
        self.absorption_window.lift()
        self.absorption_window.focus_force()
        self.absorption_window.minsize(400, 300)  # Set minimum size of the window

    def get_selected_lines(self):
        self.absorption_lines = [var.get() for var in self.check_vars_absorption if var.get()]
        self.absorption_window.destroy()

        # Call FitParamsWindow
        fit_window = FitParamsWindow(self.selected_lines, self.broad_wings_lines, self.triple_gauss_lines, self.absorption_lines)
        self.params_dict, self.params_range_dict, self.amps_ratio_dict = fit_window.run()

        # Close the LineSelector window
        self.quit()

        # Close the LineSelector window
        # self.destroy()

    def run(self):
        self.mainloop()
        # self.destroy()
        return self.selected_lines, self.selected_fitting_method, self.broad_wings_lines, self.double_gauss_lines, self.triple_gauss_lines, self.absorption_lines, self.fitting_functions_choices

# TESTING:
if __name__ == "__main__":
    elements_dict = {'Hydrogen': {'H delta': np.array([4155.7245804]),
                      'H gamma': np.array([4397.5675503]),
                      'H beta': np.array([4925.2728203]),
                      'H alpha': np.array([6649.0965307])},
                     'Oxygen': {'[OI] 6300': np.array([6383.1472548]),
                      '[OII]&[OII] 3726&3729': np.array([3775.0576483, 3777.8835556]),
                      '[OIII] 4363': np.array([4420.6103428]),
                      '[OIII] 4959': np.array([5024.139061]),
                      '[OIII] 5007': np.array([5072.6960488]),
                      '[OIII]&[OIII]&HeI 4959&5007&5015': np.array([5024.139061  , 5072.6960488 , 5081.64647713]),
                      '[OIII]&HeI 5007&5015': np.array([5072.6960488 , 5081.64647713])},
                     'Nitrogen': {'[NII] 5755': np.array([5830.3228088]),
                      '[NII]&H&[NII] 6548&alpha&6583': np.array([6634.1364408, 6649.0965307, 6669.9819101])},
                     'Sulphur': {'[SII]&[SII] 6716&6731': np.array([6804.7847784, 6819.3599777]),
                      '[SIII] 6312': np.array([6395.058606]),
                      '[SIII] 9069': np.array([9187.845057]),
                      '[SIII] 9531': np.array([9655.892284])},
                     'Helium': {'HeII 4686': np.array([4747.3115613])},
                     'Argon': {'[ArIII] 7136': np.array([7229.663486]),
                      '[ArIV] 4711': np.array([4773.3220629]),
                      '[ArIV] 4740': np.array([4802.5534911]),
                      '[ArIV]&HeI 4711&4713': np.array([4773.3220629 , 4775.13307446])},
                     'Iron': {'[FeIII] 4734': np.array([4796.1926675]),
                      '[FeIII] 4755': np.array([4817.3717792]),
                      '[FeIII] 4770': np.array([4832.3318691]),
                      '[FeIII] 4778': np.array([4840.7285614])}}
    selector = LineSelector(elements_dict)
    selected_lines, fitting_method, broad_wing_lines, double_gauss_lines, triple_gauss_lines, absorption_lines, fitting_functions_choices = selector.run()
    print(selected_lines, fitting_method, broad_wing_lines, double_gauss_lines, triple_gauss_lines, absorption_lines, fitting_functions_choices)





