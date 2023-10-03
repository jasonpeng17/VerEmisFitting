### GUI for users to select lines to be fitted 

import tkinter as tk
from tkinter import ttk
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
from params_gui import *
from line_fitting_prelim import *

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
    def __init__(self, emission_lines):
        super().__init__()

        # Font for all labels, buttons, etc.
        self.font_tuple = ("Helvetica", 14, "bold")  

        self.emission_lines = emission_lines
        self.check_vars = [tk.StringVar(value="") for line in emission_lines]
        self.fitting_methods = ["Free fitting", "Fix velocity centroid", "Fix velocity width", "Fix velocity centroid and width"]
        self.selected_method = tk.StringVar()
        self.selected_method.set(self.fitting_methods[0]) 

        # Add a label at the top of the window
        self.label = ttk.Label(self, text="Select lines for fitting", font=self.font_tuple)
        self.label.grid(row=0, column=0, columnspan=5)

        # Create checkboxes
        for i in range(len(self.emission_lines)):
            check = tk.Checkbutton(self, text=self.emission_lines[i], variable=self.check_vars[i], 
                                   onvalue=self.emission_lines[i], offvalue="", command=self.update_selected_lines)
            check.grid(row=(i//5) + 1, column=i%5, sticky='w')

        # Create OptionMenu for fitting options
        self.option_menu = tk.OptionMenu(self, self.selected_method, *self.fitting_methods)
        self.option_menu.grid(row=(i//5) + 2, column=0, columnspan=5)

        # Add button to finalize selection
        self.button = tk.Button(self, text="Confirm", command=self.select_broad_wings, font=self.font_tuple)
        self.button.grid(row=(i//5) + 3, column=0, columnspan=5)

    def update_selected_lines(self):
        self.selected_lines = [var.get() for var in self.check_vars if var.get()]

    def select_broad_wings(self):
        self.update_selected_lines()
        self.selected_fitting_method = self.selected_method.get()

        # create a new window to select lines for fitting "broad wings"
        self.broad_wings_window = tk.Toplevel(self)
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

                check = tk.Checkbutton(self.broad_wings_window, variable=var, onvalue=line, offvalue="")
                check.grid(row=row_i, column=0, sticky='w')
                self.checks_broad_wings.append(check)

                label = tk.Label(self.broad_wings_window, text=line)
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

                    check = tk.Checkbutton(self.broad_wings_window, variable=var, onvalue=line, offvalue="")
                    check.grid(row=row_i, column=0, sticky='w')
                    self.checks_broad_wings.append(check)

                    label = tk.Label(self.broad_wings_window, text=line)
                    label.grid(row=row_i, column=1, sticky='w')

                    # increase the row number by 1
                    row_i += 1

        tk.Button(self.broad_wings_window, text="Confirm", command=self.select_double_gaussians).grid(row=len(self.checks_broad_wings) + 1, column=0, columnspan=10)

        # Raise the window to the top
        self.broad_wings_window.lift()
        self.broad_wings_window.focus_force()
        self.broad_wings_window.minsize(500, 300)  # Set minimum size of the window

    def select_double_gaussians(self):
        self.broad_wings_lines = [var.get() for var in self.check_vars_broad_wings if var.get()]
        self.broad_wings_window.destroy()

        if not self.broad_wings_lines:
            # Skip directly to selecting absorption lines if no broad wings are selected for double Gaussian fitting
            self.select_absorption()
            return

        # Create a new window to select lines for fitting "double Gaussian"
        self.double_gauss_window = tk.Toplevel(self)
        self.double_gauss_window.title("Select lines that require 'double Gaussian' fitting")
        self.check_vars_double_gauss = [tk.StringVar(value="") for line in self.broad_wings_lines]

        for i in range(len(self.broad_wings_lines)):
            check = tk.Checkbutton(self.double_gauss_window, variable=self.check_vars_double_gauss[i],
                                   onvalue=self.broad_wings_lines[i], offvalue="")
            check.grid(row=i, column=0, sticky='w')
            label = tk.Label(self.double_gauss_window, text=self.broad_wings_lines[i])
            label.grid(row=i, column=1, sticky='w')

        tk.Button(self.double_gauss_window, text="Confirm", command=self.select_triple_gaussians).grid(row=i+1, column=0, columnspan=10)

        # Raise the window to the top
        self.double_gauss_window.lift()
        self.double_gauss_window.focus_force()
        self.double_gauss_window.minsize(500, 300)  # Set minimum size of the window

    def select_triple_gaussians(self):
        self.double_gauss_lines = [var.get() for var in self.check_vars_double_gauss if var.get()]
        self.double_gauss_window.destroy()

        if ((len(self.broad_wings_lines) == len(self.double_gauss_lines)) and (len(self.double_gauss_lines) >= 1) and (len(self.broad_wings_lines) >= 1)):
            # Skip directly to selecting absorption lines if only one broad wings line is selected and is selected as the "double Gaussian" one
            self.select_absorption()
            return

        # Create a new window to select lines for fitting "triple Gaussian"
        self.triple_gauss_window = tk.Toplevel(self)
        self.triple_gauss_window.title("Select lines that require 'triple Gaussian' fitting")
        self.check_vars_triple_gauss = [tk.StringVar(value="") for line in self.broad_wings_lines]

        for i in range(len(self.broad_wings_lines)):
            check = tk.Checkbutton(self.triple_gauss_window, variable=self.check_vars_triple_gauss[i],
                                   onvalue=self.broad_wings_lines[i], offvalue="")
            check.grid(row=i, column=0, sticky='w')
            label = tk.Label(self.triple_gauss_window, text=self.broad_wings_lines[i])
            label.grid(row=i, column=1, sticky='w')

        tk.Button(self.triple_gauss_window, text="Confirm", command=self.select_absorption).grid(row=i+1, column=0, columnspan=10)

        # Raise the window to the top
        self.triple_gauss_window.lift()
        self.triple_gauss_window.focus_force()
        self.triple_gauss_window.minsize(500, 300)  # Set minimum size of the window


    def select_absorption(self):
        if not hasattr(self, 'broad_wings_lines'):
            self.broad_wings_lines = []

        if hasattr(self, 'check_vars_double_gauss'):
            self.double_gauss_lines = [var.get() for var in self.check_vars_double_gauss if var.get()]
            self.double_gauss_window.destroy()
            if self.double_gauss_lines:
                self.double_gauss_broad = messagebox.askquestion("Confirmation", "Are there any broad wings in the lines that require 'double Gaussian' fitting?")
            else:
                self.double_gauss_broad = 'no'
        else:
            self.double_gauss_lines = []
            self.double_gauss_broad = 'no'

        if hasattr(self, 'check_vars_triple_gauss'):
            self.triple_gauss_lines = [var.get() for var in self.check_vars_triple_gauss if var.get()]
            self.triple_gauss_window.destroy()
            if self.triple_gauss_lines:
                self.triple_gauss_broad = messagebox.askquestion("Confirmation", "Are there any broad wings in the lines that require 'triple Gaussian' fitting?")
            else:
                self.triple_gauss_broad = 'no'
        else:
            self.triple_gauss_lines = []
            self.triple_gauss_broad = 'no'

        # create a new window to select lines for fitting "absorption"
        self.absorption_window = tk.Toplevel(self)
        self.absorption_window.title("Select lines that have 'absorption troughs'")
        self.check_vars_absorption = [tk.StringVar(value="") for line in self.selected_lines]

        for i in range(len(self.selected_lines)):
            check = tk.Checkbutton(self.absorption_window, variable=self.check_vars_absorption[i],
                                   onvalue=self.selected_lines[i], offvalue="")
            check.grid(row=i, column=0, sticky='w')
            label = tk.Label(self.absorption_window, text=self.selected_lines[i])
            label.grid(row=i, column=1, sticky='w')

        tk.Button(self.absorption_window, text="Confirm", command=self.get_selected_lines).grid(row=i+1, column=0, columnspan=10)

        # Raise the window to the top
        self.absorption_window.lift()
        self.absorption_window.focus_force()
        self.absorption_window.minsize(400, 300)  # Set minimum size of the window

    def get_selected_lines(self):
        self.absorption_lines = [var.get() for var in self.check_vars_absorption if var.get()]
        self.absorption_window.destroy()

        # Close the LineSelector window
        self.destroy()

        # Call FitParamsWindow
        fit_window = FitParamsWindow(self.selected_lines, self.broad_wings_lines, self.triple_gauss_lines, self.absorption_lines)
        self.params_dict, self.params_range_dict, self.amps_ratio_dict = fit_window.run()

        # print("Parameters:", self.params_dict)
        # print("Parameter ranges:", self.params_range_dict)
        # print("amplitude ratios:", self.amps_ratio_dict)
        # embed()

        # Close the LineSelector window
        self.quit()

    def run(self):
        self.mainloop()
        # self.destroy()
        return self.selected_lines, self.selected_fitting_method, self.broad_wings_lines, self.double_gauss_lines, self.triple_gauss_lines, self.absorption_lines

# TESTING:
if __name__ == "__main__":
    emission_lines = ['[OII]&[OII] 3726&3729', '[OIII] 4363', '[OIII] 4959', '[OIII] 5007', '[OIII]&HeI 5007&5015', 'H beta', 'H gamma', 'H delta', 
                      'H 16-2', 'H 12-2', 'H 11-2', 'He I 5015', 'He II 4686', '[NII]&H&[NII] 6548&alpha&6583']
    selector = LineSelector(emission_lines)
    selected_lines, fitting_method, broad_wing_lines, double_gauss_lines, triple_gauss_lines, absorption_lines = selector.run()
    print(selected_lines, fitting_method, broad_wing_lines, double_gauss_lines, triple_gauss_lines, absorption_lines)





