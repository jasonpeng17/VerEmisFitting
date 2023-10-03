# this class is for reddening correction: either for intrinsic or MW reddening correction


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import interpolate
import matplotlib as mp

class extinction:
    
    def __init__(self):
        self.R_v = None
        self.law = None
        self.E_BV = None
            
    def find_fm_unred_k(self,wave,print_result=False,return_k=True,Rv=3.1):
    
        if wave < 2700:
            # for UV part (<2700A): parameterized
            # x0 and gamma are in micron^-1
            # x in micron^-1
            wave_micron = wave/10000.
            x = 1.0/ wave_micron

            x0 = 4.596
            gamma = 0.99
            c3 = 3.23
            c4 = 0.41
            c2 = -0.824 + 4.717/Rv
            c1 = 2.030 - 3.007*c2
            self.FitzParams = [x0, gamma, c1, c2, c3, c4]
            D = (x**2) / ((x**2 - x0**2)**2 + ((x*gamma)**2))
            if x >= 5.9:
                F = 0.5392 * ((x-5.9)**2) + 0.05644*((x-5.9)**3)
            else:
                F = 0

            k = c1 + c2*x + c3*D + c4*F + Rv
            part = "uv"

        else:
            # for optical/IR part: cubic spline
            # wavelength in Angstrom
            # 1e8 A is to simulate the case for infinity
            wavelength = np.array([2600,2700,4110,4670,5470,6000,12200,26500,1e8])
            klambda = np.array([6.591,6.265,4.315,3.806,3.055,2.688,0.829,0.265,0.0])

            #cubic spline: k = 3
            tck,fp,ier,msg = interpolate.splrep(wavelength,klambda,k=3,full_output=1)

            k = interpolate.splev(wave,tck,der=0)
            part = "opir"

        if print_result:
            print ("k_lambda = ", k)
            print ("wavelength (A) = ", wave)
            print ("part = ", part, " ; k_lambda = ", k)
            if part == "opir":
                print ("=========")
                print ("fp = ", fp)
                print ("ier = ", ier)
                print ("msg = ", msg)
                print ("")
        if return_k:
            return np.float64(k),part
        
    def find_fmcorr_flux(self,f_obs,wave,return_result=True,print_result=False):

        k_lam,part = self.find_fm_unred_k(wave,print_result=print_result,return_k=True)
        A_lam = k_lam*self.E_BV
        corr_fac = (10**(0.4*A_lam))
        f_int = f_obs * corr_fac
        if print_result:
            print ("k_lambda = ", k_lam)
            print ("flux correction factor = ", corr_fac)
            print ("f_obs, f_int = ", f_obs, f_int)

        if return_result:
            return f_int,k_lam,corr_fac
        
    def find_fmobs_flux(self, f_int, wave, return_result=True):
        k_lam,part = self.find_fm_unred_k(wave,print_result=False,return_k=True)
        A_lam = k_lam*self.E_BV
        corr_fac = (10**(0.4*A_lam))
        f_obs = f_int / corr_fac
        
        if return_result:
            return f_obs,k_lam,corr_fac
    
    def plot_fm_unred(self,Rv=3.1):
    
        wavearr = np.arange(910,22000,1)
        wave_inv_arr = (wavearr/1e4)**(-1)

        karr = np.array([])
        parr = np.array([])
        for i in np.arange(len(wavearr)):
            ktemp, ptemp = self.find_fm_unred_k(wavearr[i],print_result=False,return_k=True,Rv=Rv)
            karr = np.append(karr,[ktemp])
            parr = np.append(parr,[ptemp])
        garr = karr/Rv
        fig = plt.figure()
        plt.plot(wave_inv_arr,garr,'k')
        plt.xlabel('x (1/wave(micron)')
        plt.ylabel(r'$A(\lambda) / A(V)$')
        plt.title('Fitz+1998')
        fig.show()


#         fig = plt.figure()
#         ax = fig.add_subplot(111)

#         wavelength = np.array([2600,2700,4110,4670,5470,6000,12200,26500,1e8])
#         klambda = np.array([6.591,6.265,4.315,3.806,3.055,2.688,0.829,0.265,0.0])

#         ax.plot(wavelength,klambda,"bx",markersize=9,label="Fitzpatrick 1999 (Table 3)")
#         ax.plot(wave_inv_arr,karr,"r-")

#         ax.set_xlabel(r'$1/ \lambda \ (\mu m^{-1})$', fontsize=16)
#         ax.set_ylabel(r'$k(\lambda)$', fontsize=16)
#         ax.set_xlim(900,22000)
#         ax.set_xscale('log')
#         ax.xaxis.set_tick_params(which='major',length=10,labelsize=14)
#         ax.xaxis.set_tick_params(which='minor',length=3)
#         ax.yaxis.set_tick_params(which='major',length=10,labelsize=14)

#         ax.get_xaxis().set_major_formatter(mp.ticker.ScalarFormatter())
#         ax.legend(loc="upper right")
#         fig.tight_layout()
#         fig.show()

    def find_gordan_aa(self,wave):
        wave_micron = wave/1e4
        gordan2003_curves_txt = os.getcwd() + '/gordan2003_curves.txt'
        w, x, aa = np.loadtxt(gordan2003_curves_txt , usecols=(0,1,2), unpack=True, skiprows=1)
#         w = w * 1e4
        # Class returns a function whose call method uses interpolation to find values of new points.
        func = interpolate.interp1d(w, aa, kind = 'linear', fill_value="extrapolate")
        g = func(wave_micron)

        return g

    # Return the corrected flux
    def find_gordan_flux(self,f_obs, wave, AV):
        wave_micron = wave/1e4
        gordan2003_curves_txt = os.getcwd() + '/gordan2003_curves.txt'
        w, x, aa = np.loadtxt(gordan2003_curves_txt , usecols=(0,1,2), unpack=True, skiprows=1)
#         w = w * 1e4
        # Class returns a function whose call method uses interpolation to find values of new points.
        func = interpolate.interp1d(w, aa, kind = 'linear', fill_value="extrapolate")
        g = func(wave_micron)
        fcorr = 10 ** (0.4 * g * AV)
        f_int = f_obs * fcorr

        return f_int
    # Find the extinction from any Balmer decrement; wavelengths are in Angstrom
    def find_gordan_av(self,bratio_obs, bratio_int, wave_red, wave_blue):

        # bratio_obs = the measured Balmer ratio
        # bratio_int = the intrinsic Balmer ratio
        gordan2003_curves_txt = os.getcwd() + '/gordan2003_curves.txt'
        w, x, aa = np.loadtxt(gordan2003_curves_txt , usecols=(0,1,2), unpack=True, skiprows=1)
#         w = w * 1e4
        # Class returns a function whose call method uses interpolation to find values of new points.
        func = interpolate.interp1d(w, aa, kind = 'linear', fill_value="extrapolate")
        gred = func(wave_red/1e4)
        gblue = func(wave_blue/1e4)

        av = 2.5 / (gred - gblue) * np.log10(bratio_int / bratio_obs)

        return av
    
    def plot_gordan_unred(self):
#         plt.plot(w,aa,'k')
#         plt.xlabel('Wave (Angstrom)')
#         plt.ylabel('A(lambda) / A(V)')
#         plt.title('SMC bar (Gordan+2003)')

#         y = np.array([4000, 5000, 5400, 5500, 5600, 6000,7000]) 
#         g = find_gordan_aa(y)
#         plt.plot(y,g,'bo')

#         fig.show()
        wavearr = np.arange(910,22000,1)
        wave_inv_arr = (wavearr/1e4)**(-1)
        xiarr = np.array([])

        for i in np.arange(len(wavearr)):
            xitemp = self.find_gordan_aa(wavearr[i])
            xiarr = np.append(xiarr,[xitemp])
        fig = plt.figure()
        plt.plot(wave_inv_arr,xiarr,'k')
        plt.xlabel('x (1/wave(micron)')
        plt.ylabel(r'$A(\lambda) / A(V)$')
        plt.title('SMC (Gordan+2003 Table 4)')
        fig.show()
        
    def find_pei_xi_smc(self,wave):
        pei1992_curves_txt = os.getcwd() + '/pei1992_curves.txt'
        w_inv, e_lamv = np.loadtxt(pei1992_curves_txt, usecols=(0,1), unpack=True, skiprows=1)
        w = (1./w_inv)
        R_v = 2.93
        xi = (e_lamv + R_v)/(1 + R_v)
        gi = 1.-(e_lamv/(-R_v))
        # Class returns a function whose call method uses interpolation to find values of new points.
        func_xi = interpolate.interp1d(w, xi, kind = 'linear', fill_value="extrapolate")
        func_gi = interpolate.interp1d(w, gi, kind = 'linear', fill_value="extrapolate")
        # A(lambda)/A(B)
        xi_value = func_xi(wave/1e4)
        # A(lambda)/A(V)
        gi_value = func_gi(wave/1e4)

        return xi_value,gi_value
    
    def find_pei_ab(self,bratio_obs, bratio_int, wave_red, wave_blue):

        # bratio_obs = the measured Balmer ratio
        # bratio_int = the intrinsic Balmer ratio

        pei1992_curves_txt = os.getcwd() + '/pei1992_curves.txt'
        w_inv, e_lamv = np.loadtxt(pei1992_curves_txt, usecols=(0,1), unpack=True, skiprows=1)
        w = (1./w_inv)
        R_v = 2.93
        xi = (e_lamv + R_v)/(1 + R_v)
        # Class returns a function whose call method uses interpolation to find values of new points.
        func = interpolate.interp1d(w, xi, kind = 'linear', fill_value="extrapolate")
        xired = func(wave_red/1e4)
        xiblue = func(wave_blue/1e4)

        ab = 2.5 / (xired - xiblue) * np.log10(bratio_int / bratio_obs)

        return ab
    
    def find_pei_flux(self,fobs, wave, ab):
        pei1992_curves_txt = os.getcwd() + '/pei1992_curves.txt'
        w_inv, e_lamv = np.loadtxt(pei1992_curves_txt, usecols=(0,1), unpack=True, skiprows=1)
        w = (1./w_inv)
        R_v = 2.93
        xi = (e_lamv + R_v)/(1 + R_v)
        # Class returns a function whose call method uses interpolation to find values of new points.
        func = interpolate.interp1d(w, xi, kind = 'linear', fill_value="extrapolate")
        xi = func(wave/1e4)
        fcorr = 10 ** (0.4 * xi * ab)

        return fcorr
    
    def plot_pei_unred(self):
        wavearr = np.arange(910,22000,1)
        wave_inv_arr = (wavearr/1e4)**(-1)

        xiarr = np.array([])
        giarr = np.array([])
        
        for i in np.arange(len(wavearr)):
            xitemp = self.find_pei_xi_smc(wavearr[i])[0]
            gitemp = self.find_pei_xi_smc(wavearr[i])[1]
            xiarr = np.append(xiarr,[xitemp])
            giarr = np.append(giarr,[gitemp])
        fig = plt.figure()
#         plt.plot(wave_inv_arr,giarr,'k',label =r'$A(\lambda)/A(V)$')
        plt.plot(wave_inv_arr,xiarr,'r',label =r'$A(\lambda)/A(B)$')
        plt.xlabel('x (1/wave(micron)')
        plt.ylabel(r'$A(\lambda) / A(B)$')
        plt.title('SMC (Pei+1992 Table 1)')
        plt.legend()
        fig.show()
        
        
    
    def find_pei_ab_2(self,bratio_obs, bratio_int, wave_red, wave_blue):
        wave_red_micron = wave_red/1e4
        wave_blue_micron = wave_blue/1e4
        
        a_i = np.array([185.,27.,0.005,0.010,0.012,0.030])
        lam_i = np.array([0.042,0.08,0.22,9.7,18.,25.])
        b_i = np.array([90.,5.50,-1.95,-1.95,-1.80,0.00])
        n_i = np.array([2.0,4.0,2.0,2.0,2.0,2.0])
        xi_red = np.sum(a_i/((wave_red_micron/lam_i)**n_i + (lam_i/wave_red_micron)**n_i + b_i))
        xi_blue = np.sum(a_i/((wave_blue_micron/lam_i)**n_i + (lam_i/wave_blue_micron)**n_i + b_i))
        ab = 2.5 / (xi_red - xi_blue) * np.log10(bratio_int / bratio_obs)
        
        return ab

    def find_pei_flux_2(self,fobs,wave,ab):
        wave_micron = wave/1e4
        a_i = np.array([185.,27.,0.005,0.010,0.012,0.030])
        lam_i = np.array([0.042,0.08,0.22,9.7,18.,25.])
        b_i = np.array([90.,5.50,-1.95,-1.95,-1.80,0.00])
        n_i = np.array([2.0,4.0,2.0,2.0,2.0,2.0])
        xi = np.sum(a_i/((wave_micron/lam_i)**n_i + (lam_i/wave_micron)**n_i + b_i))
        
        fcorr = 10 ** (0.4 * xi * ab)
        fint = fobs * fcorr
        return fint, fcorr, xi
    
    def find_pei_xi_smc_2(self,wave):
        wave_micron = wave/1e4
        a_i = np.array([185.,27.,0.005,0.010,0.012,0.030])
        lam_i = np.array([0.042,0.08,0.22,9.7,18.,25.])
        b_i = np.array([90.,5.50,-1.95,-1.95,-1.80,0.00])
        n_i = np.array([2.0,4.0,2.0,2.0,2.0,2.0])
        xi = np.sum(a_i/((wave_micron/lam_i)**n_i + (lam_i/wave_micron)**n_i + b_i))
    
        return xi
    
    def plot_pei_unred_2(self):
    
        wavearr = np.arange(910,22000,1)
        wave_inv_arr = (wavearr/1e4)**(-1)

        xiarr = np.array([])
        
        for i in np.arange(len(wavearr)):
            xitemp = self.find_pei_xi_smc_2(wavearr[i])
            xiarr = np.append(xiarr,[xitemp])
  


#         fig = plt.figure()
#         ax = fig.add_subplot(111)

#         wavelength = np.array([2600,2700,4110,4670,5470,6000,12200,26500,1e8])
#         klambda = np.array([6.591,6.265,4.315,3.806,3.055,2.688,0.829,0.265,0.0])

#         ax.plot(wave_inv_arr,xiarr,"r-")

#         ax.set_xlabel(r'$1/ \lambda \ (\mu m^{-1}$', fontsize=16)
#         ax.set_ylabel(r'$\xi(\lambda)$', fontsize=16)
#         ax.set_xlim(0,10)
#         ax.set_xscale('log')
#         ax.set_yscale('log')
#         ax.plot(label="Pei 1992 (Table 4)")
#         ax.xaxis.set_tick_params(which='major',length=10,labelsize=14)
#         ax.xaxis.set_tick_params(which='minor',length=3)
#         ax.yaxis.set_tick_params(which='major',length=10,labelsize=14)

#         ax.get_xaxis().set_major_formatter(mp.ticker.ScalarFormatter())
#         ax.legend(loc="upper right")
#         fig.tight_layout()
#         fig.show()

        fig = plt.figure()
        plt.plot(wave_inv_arr,xiarr,'k')
        plt.xlabel('x (1/wave(micron)')
        plt.ylabel(r'$A(\lambda) / A(B)$')
        plt.title('SMC (Pei+1992 Table 4) ')
        fig.show()
    
    def find_calz_k(self,wave,Rv=4.05,print_result=False,return_k=True):
        # Calzetti curve wants micron, 1 um = 10000A
        wave_micron = wave/10000.
        #print ("wave_micron = ", wave_micron)
        x = 1.0/wave_micron
        err_flag = "none"

        if ((wave >= 912) and (wave < 6300)):
            k = 2.659 * (-2.156 + 1.509*x - 0.198*(x**2) + 0.011*(x**3)) + Rv
            if wave < 1200:
                err_flag = "912-1200"
        elif ((wave >= 6300) and (wave <= 22000)):
            k = 2.659 * (-1.857 + 1.040*x) + Rv
        else:
            err_flag = "out_range"
            k = -99

        if print_result:
            print ("wavelength (A) = ", wave)
            print ("k_lambda = ", k)
        if return_k:
            return k,err_flag
        
    def find_calzcorr_flux(f_obs,wave,eBV_s=-1,eBV=-1,return_result=True,print_result=False):
        err_flag = "none"
        if (eBV < 0) and (eBV_s < 0):
            if return_result:
                k_lam = -99
                err_flag = "no_E(B-V)"
            else:
                sys.exit("No input in E(B-V) or Es(B-V)!")
        else:
            k_lam,err_flag = self.find_calz_k(wave,print_result=False,return_k=True)

        if eBV_s >= 0:
            EEBV = eBV_s
            eBV_choice = "Es(B-V)"
        elif eBV >= 0:
            EEBV = eBV*0.44
            eBV_choice = "E(B-V)"

        if k_lam != -99:
            corr_fac = (10**(0.4*EEBV*k_lam))
            f_int = f_obs * corr_fac
        else:
            corr_fac = 1
            f_int = f_obs

        if print_result:
            print ("Use ", eBV_choice)
            print ("k_lambda = ", k_lam)
            print ("flux correction factor = ", corr_fac)
            print ("f_obs, f_int = ", f_obs,f_int)
        if return_result:
            return f_int,k_lam,corr_fac,err_flag
        
    
        

        
    
    
    
