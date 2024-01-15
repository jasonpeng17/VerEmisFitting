### GUI for users to select fitting window, local continuum regions; and also mask lines

import os
import subprocess
import requests
import time
from bokeh.plotting import show, output_file, save
from bokeh.models import Span, BoxAnnotation
from bokeh.client import pull_session
from bokeh.plotting.figure import figure
from bokeh.layouts import column
from bokeh.plotting.figure import Figure
from bokeh.models import Column
from termcolor import colored
import numpy as np

class FittingWindow:
    def __init__(self, wave, flux, fits_name = None, line_name = None):
        self.wave = wave # input wavelength arr
        self.flux = flux # input flux arr
        self.wave_lolim = np.nanmin(self.wave) # wavelength lower limit 
        self.wave_uplim = np.nanmax(self.wave) # wavelength upper limit
        self.yrange = [np.nanmin(self.flux) * 0.8, np.max(self.flux) * 1.2] # flux range for plotting
        self.plot_width = 1200 # plot width for bokeh figure
        self.plot_height = 600 # plot height for bokeh figure
        self.fits_name = fits_name # fits name 
        self.line_name = line_name # name of the selected line(s)

    # @staticmethod
    def ensure_bokeh_server(self):
        # Check if Bokeh server is running
        try:
            response = requests.get("http://localhost:5006/")
            if response.status_code == 200:
                print("Bokeh server is already running.")
                return True
        except requests.ConnectionError:
            print("Bokeh server is not running. Starting it now...")
        
        # Start Bokeh server in a subprocess
        self.bokeh_process = subprocess.Popen(["bokeh", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for the server to start
        time.sleep(1)
        print("Bokeh server started.")

    def stop_bokeh_server(self):
        # stop/terminate the bokeh server
        if hasattr(self, 'bokeh_process'):
            self.bokeh_process.terminate()
            self.bokeh_process = None

    def start_bokeh(self):
        self.ensure_bokeh_server()
        # session = pull_session(session_id='fitting_window', url='http://localhost:5006')
        session = pull_session()
        # self.logger.info("Enabling BOKEH plots")
        p = figure()
        c = column(children=[p])
        session.document.clear()
        session.document.add_root(c)
        # self.context.bokeh_session = session
        session.show(c)
        return session 

    def bokeh_plot(self, plot, session):
        new_figure = session.document.select_one(selector=dict(type=Figure))
        layout = session.document.select_one(selector=dict(type=Column))
        layout.children.remove(new_figure)
        layout.children.insert(0, plot)
        session.push()

    # copy from https://github.com/Keck-DataReductionPipelines/KCWI_DRP/blob/1954e630ac375c187bd3a897f87ccb0f970cf014/kcwidrp/core/kcwi_plotting.py
    def set_plot_lims(self, fig, xlim=None, ylim=None):
        """Set bokeh figure plot ranges"""
        if xlim:
            fig.x_range.start = xlim[0]
            fig.x_range.end = xlim[1]
        if ylim:
            fig.y_range.start = ylim[0]
            fig.y_range.end = ylim[1]

    def find_local_cont_file(self, line_name):
        # Look for local continum region file
        # first in local directory
        cont_file_path = os.path.join(os.getcwd(), f'cont_dir/{self.fits_name}')
        if not os.path.exists(cont_file_path):
            os.makedirs(cont_file_path)
        self.local_contfile = os.path.join(cont_file_path, f"{line_name.split(' ')[0]}_{line_name.split(' ')[1]}.cont") # assume the local cont file is in the local folder cont_dir
        if not os.path.exists(self.local_contfile):
            contfile = None
        else:
            contfile = self.local_contfile
        # Header for local cont file
        conthdr = []
        # local cont regions
        cont_fit = []
        # if we found a mask file, read it in
        if contfile is None:
            print("No local continuum region file found")
        else:
            print("Using local continuum region file %s" % contfile)
            with open(contfile) as ctf:
                cont_str = ctf.readlines()
            # parse file into mask list
            for cts in cont_str:
                cts = cts.strip()
                # Collect header lines
                if cts.startswith('#'):
                    conthdr.append(cts)
                    continue
                # Skip blank lines
                if len(cts.strip()) < 1:
                    continue
                try:
                    data = cts.split('#')[0]
                    # Parse comment on line
                    if '#' in cts:
                        comm = cts.split('#')[1].lstrip()
                    else:
                        comm = ''
                    # Parse line mask range
                    ct = [float(v) for v in data.split()]
                except ValueError:
                    print("bad line: %s" % cts)
                    continue
                # Only collect good lines
                if len(ct) == 4 and (ct[0] < ct[2] < ct[3] < ct[1]) and (ct[2] < ct[3]):
                    ctdict = {'x1': ct[0], 'x2': ct[1], 'x3': ct[2], 'x4': ct[3], 'com': comm}
                    cont_fit.append(ctdict)
                else:
                    print("bad line: %s" % cts)
        return cont_fit, conthdr
    
    # based on https://github.com/Keck-DataReductionPipelines/KCWI_DRP/blob/1954e630ac375c187bd3a897f87ccb0f970cf014/kcwidrp/primitives/MakeInvsens.py
    def find_local_lmsk_file(self, line_name):
        # Look for mask data file
        # first in local directory
        lmsk_file_path = os.path.join(os.getcwd(), f'lmsk_dir/{self.fits_name}')
        if not os.path.exists(lmsk_file_path):
            os.makedirs(lmsk_file_path)
        self.local_lmfile = os.path.join(lmsk_file_path, f"{line_name.split(' ')[0]}_{line_name.split(' ')[1]}.lmsk") # assume the lmsk file is in the local folder lmsk_dir
        if not os.path.exists(self.local_lmfile):
            lmfile = None
        else:
            lmfile = self.local_lmfile
        # Header for line mask file
        lmhdr = []
        # Line masks
        lmasks = []
        # if we found a mask file, read it in
        if lmfile is None:
            print("No line mask file found")
        else:
            print("Using line mask file %s" % lmfile)
            with open(lmfile) as lmf:
                lmasks_str = lmf.readlines()
            # parse file into mask list
            for lmws in lmasks_str:
                lmws = lmws.strip()
                # Collect header lines
                if lmws.startswith('#'):
                    lmhdr.append(lmws)
                    continue
                # Skip blank lines
                if len(lmws.strip()) < 1:
                    continue
                try:
                    data = lmws.split('#')[0]
                    # Parse comment on line
                    if '#' in lmws:
                        comm = lmws.split('#')[1].lstrip()
                    else:
                        comm = ''
                    # Parse line mask range
                    lm = [float(v) for v in data.split()]
                except ValueError:
                    print("bad line: %s" % lmws)
                    continue
                # Only collect good lines
                if len(lm) == 2 and lm[0] < lm[1]:
                    lmdict = {'w0': lm[0], 'w1': lm[1], 'com': comm}
                    lmasks.append(lmdict)
                else:
                    print("bad line: %s" % lmws)
        return lmasks, lmhdr

    def save_cont_list(self, cont, conthdr):
        # Write out a local copy of line continuum region so user can edit
        if len(cont) > 0:
            with open(self.local_contfile, 'w') as contf:
                for conth in conthdr:
                    contf.write(conth)
                for ct in cont:
                    if len(ct['com']) == 0:
                        contf.write("%.2f %.2f %.2f %.2f\n" % (ct['x1'], ct['x2'], ct['x3'], ct['x4']))
                    else:
                        contf.write("%.2f %.2f %.2f %.2f # %s\n" % (ct['x1'], ct['x2'], ct['x3'], ct['x4'], ct['com']))

    def save_lmsk_list(self, lmasks, lmhdr):
        # Write out a local copy of line masks so user can edit
        if len(lmasks) > 0:
            # lmsk_file_path = os.path.join(os.getcwd(), f'lmsk_dir')
            # if not os.path.exists(lmsk_file_path):
            #     os.makedirs(lmsk_file_path)
            # local_lmfile = os.path.join(lmsk_file_path, f'{self.fits_name}.lmsk')
            with open(self.local_lmfile, 'w') as lmf:
                for lmh in lmhdr:
                    lmf.write(lmh)
                for lm in lmasks:
                    if len(lm['com']) == 0:
                        lmf.write("%.2f %.2f\n" % (lm['w0'], lm['w1']))
                    else:
                        lmf.write("%.2f %.2f # %s\n" % (lm['w0'], lm['w1'], lm['com']))

    def plot_with_window(self, p, x0, x1, x2, x3, x4):
        # plot spectrum with central wavelength, fitting window, and local continuum regions
        # Plot the spectrum
        p.line(self.wave, self.flux, line_color='black', line_width=1, legend_label="Spectrum")

        # Plot central wavelength, fitting window, and local continuum regions
        p.line([x0, x0], self.yrange, line_color='blue', line_dash='dashed', line_width=3, legend_label="Center of Fitting Window")
        p.line([x1, x1], self.yrange, line_color='red', line_width=3, legend_label="Lower End of Fitting Window")
        p.line([x2, x2], self.yrange, line_color='red', line_width=3, legend_label="Upper End of Fitting Window")
        p.line([x3, x3], self.yrange, line_color='red', line_dash='dashed', line_width=3, legend_label="Upper end of the Left Local Continuum Region")
        p.line([x4, x4], self.yrange, line_color='red', line_dash='dashed', line_width=3, legend_label="Lower end of the Right Local Continuum Region")

        # display legend in top left corner (default is top right corner)
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.background_fill_alpha = 0.2
        # set the x and y ranges for figure
        self.set_plot_lims(p, xlim=[self.wave_lolim * 0.95, self.wave_uplim * 1.05], ylim=self.yrange)


    def run_process(self, x0, x1, x2, x3, x4, mask_lines = False):
        # Step 1: Plot the whole spectrum
        print(colored("CHECKING LINE FITTING WINDOW AND LOCAL CONTINUUM REGIONS", 'green', attrs=['bold', 'underline']))
        print(colored(f"Current line fitting window: {x1:.1f} (x1) - {x2:.1f} (x2) Angstroms (red vertical solid lines)", 'green', attrs=['bold']))
        print(colored(f"Current local continuum regions: {x1:.1f} (x1) - {x3:.1f} (x3) Angstroms and {x4:.1f} (x4) - {x2:.1f} (x2) Angstroms", 'green', attrs=['bold']))
        print(colored(f"A <cr> will accept current values or enter new values to determine a new line fitting window or local continuum regions.", 'green', attrs=['bold']))
        print(colored(f"Hover the cursor over spectrum to find wavelengths. \n", 'green', attrs=['bold']))

        done = False # initialize the done value to be False 
        self.session = self.start_bokeh() # initialize the bokeh server
        # find local continuum region file
        cont, conthdr = self.find_local_cont_file(self.line_name)
        if len(cont) == 1:
            for ct in cont:
                x1, x2, x3, x4 = ct['x1'], ct['x2'], ct['x3'], ct['x4']
        # interactively set wavelength limits
        while not done:
            # create the bokeh figure
            p = figure(title='Spectrum with Line Fitting Window', 
                       tooltips=[("x", "@x{0.0}"), ("y", "@y{0.0}")],
                       x_axis_label='Wavelength (Å)', y_axis_label='Flux',
                       plot_width=self.plot_width, 
                       plot_height=self.plot_height)

            # plot spectrum with central wavelength, fitting window, and local continuum regions
            self.plot_with_window(p, x0, x1, x2, x3, x4)
            self.bokeh_plot(p, self.session)
            input_str = input("Enter new left and right boundaries for fitting window and local continuum regions"
                              "x1: <float> x2: <float> x3: <float> x4: <float> comment: <str> (<cr> - done): ")
            if len(input_str) <= 0:
                done = True
            else:
                try:
                    ctdict = {'x1': float(input_str.split()[0]),
                              'x2': float(input_str.split()[1]),
                              'x3': float(input_str.split()[2]),
                              'x4': float(input_str.split()[3]),
                              'com': " ".join(input_str.split()[4:]).strip()}
                    x1, x2, x3, x4 = ctdict['x1'], ctdict['x2'], ctdict['x3'], ctdict['x4']
                    conthdr = ctdict['com']
                    # boundary = [float(val) for val in input_str.split()]
                    wave_check = np.array([(self.wave_lolim > x_p or self.wave_uplim < x_p) for x_p in np.array([x1, x2, x3, x4])])
                    if wave_check.any(): # check whether any input is outside the wavelength boundary
                        invalid_indx = np.where(wave_check == True)[0][0]
                        print(f"\ninput x{invalid_indx + 1}: {boundary[invalid_indx]} is outside wavelength range, try again")
                except (IndexError, ValueError):
                    print("\nformat error, try again")
        # update region of interest
        try:
            if len(ctdict.keys()) >= 4:
                cont.clear() # clear the original local cont list
                cont.append(ctdict)
        except UnboundLocalError:
            if len(cont) == 0:
                ctdict = {'x1': x1,
                          'x2': x2,
                          'x3': x3,
                          'x4': x4,
                          'com': "default values"}
                cont.append(ctdict)
                print("\nno new local continuum regions updated and save default values")
            else:
                print("\nno new local continuum regions updated")
        boundary = np.array([x1, x2, x3, x4])
        self.save_cont_list(cont, conthdr)
        # END: interactively identify lines

        # interactively mask lines
        if mask_lines:
            # find local mask file
            lmasks, lmhdr = self.find_local_lmsk_file(self.line_name)
            print(colored("\nMASKING SHARP FEATURES: UNINTENDED ABSORPTION LINES/EMISSION LINES", 'green', attrs=['bold', 'underline']))
            print(colored("To mask, enter starting and stopping wavelengths for each line you want to mask.", 'green', attrs=['bold']))
            print(colored("Current masks are shown as vertical dashed yellow lines.", 'green', attrs=['bold']))
            print(colored("Hover the cursor over spectrum to find wavelengths.", 'green', attrs=['bold']))
            print(colored(f"Mask values are saved to this file: {self.local_lmfile}.", 'green', attrs=['bold']))
            print(colored("A <cr> with no line limits will accept current masks.", 'green', attrs=['bold']))

            done_mask = False # initialize the done_mask value to be False 
            while not done_mask:
                p = figure(title='Spectrum with Line Fitting Window', 
                           tooltips=[("x", "@x{0.0}"), ("y", "@y{0.0}")],
                           x_axis_label='Wavelength (Å)', y_axis_label='Flux',
                           plot_width=self.plot_width, 
                           plot_height=self.plot_height)

                # plot spectrum with central wavelength, fitting window, and local continuum regions
                self.plot_with_window(p, x0, x1, x2, x3, x4)
                
                # Plot masked lines from the local lmsk file
                for ml in lmasks:
                    if self.wave_lolim < ml['w0'] < self.wave_uplim and self.wave_lolim < ml['w1'] < self.wave_uplim:
                        p.line([ml['w0'], ml['w0']], self.yrange, line_color='orange', line_dash='dashed', line_width=2)
                        p.line([ml['w1'], ml['w1']], self.yrange, line_color='orange', line_dash='dashed', line_width=2)

                self.bokeh_plot(p, self.session)
                newl_list = []
                lmsk_str = input("Mask line: wavelength start stop (Ang) comment "
                                 "<float> <float> <str> (<cr> - done): ")
                if len(lmsk_str) <= 0:
                    done_mask = True
                else:
                    try:
                        newl = {'w0': float(lmsk_str.split()[0]),
                                'w1': float(lmsk_str.split()[1]),
                                'com': " ".join(lmsk_str.split()[2:]).strip()}
                        lmhdr = newl['com']
                    except ValueError:
                        print("\nbad line: %s" % lmsk_str)
                        continue
                    if self.wave_lolim < newl['w0'] < self.wave_uplim and self.wave_lolim < newl['w1'] < self.wave_uplim:
                        lmasks.append(newl)
                    else:
                        print("\nline mask outside wl range: %s" % lmsk_str)
            # END: interactively mask lines
            self.save_lmsk_list(lmasks, lmhdr)
        # Stop the server
        self.stop_bokeh_server()
        return boundary, lmasks

# TESTING:
if __name__ == "__main__":
    from astropy.io import fits
    import numpy as np
    # define the absolute path of the current working directory
    current_direc = os.getcwd() 

    # define the absolute path of the input fits files
    data_fits = current_direc + '/example_inputs/j1044+0353_addALL_icubes_wn.fits'
    err_fits = current_direc + '/example_inputs/j1044+0353_addALL_icubes_wn_err.fits'
    spec = fits.open(data_fits)[0].data # flux array
    err = fits.open(err_fits)[0].data # error array
    wave = np.load(current_direc + '/example_inputs/wave_grid.npy') # wavelength array
    redshift = 0.01287

    # initialize the class
    Fitting_Window = FittingWindow(wave, spec, fits_name = 'j1044+0353_addALL_icubes_wn')

    x0 = 5007 * (1. + redshift)  # Example central wavelength (e.g., H-alpha line)
    x1, x2 = 4970 * (1. + redshift), 5050 * (1. + redshift)  # Initial guesses for the fitting window
    x3, x4 = 4990 * (1. + redshift), 5030 * (1. + redshift)  # Initial guesses for the continuum regions

    boundary, lmasks = Fitting_Window.run_process(x0, x1, x2, x3, x4, mask_lines = True)
    print(boundary)
    print(lmasks)





