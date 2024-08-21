import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy.visualization import quantity_support
import pandas as pd

def generate_mock_lines():
    ha_pab_ratio = 15

    ha_wave = 6564.6
    ha_amp = 2.32e-19
    ha_sigma = 50
    ha_flux  = ha_amp * ha_sigma * np.sqrt(2 * np.pi)

    pab_wave = 12821.7
    pab_sigma = 25
    pab_flux  = ha_flux / ha_pab_ratio
    pab_amp = pab_flux / ha_sigma * np.sqrt(2 * np.pi)
 

    wavelength, flux = generate_blackbody(T = 3000)
    c = 299792458 # m/s
    flux_erg_aa = flux * (1e-23*1e10*c / (wavelength**2))
    
    flux_erg_aa = add_emission_line(wavelength, flux_erg_aa, ha_wave, ha_amp, ha_sigma)
    flux_erg_aa = add_emission_line(wavelength, flux_erg_aa, pab_wave, pab_amp, pab_sigma)
    # plt.plot(wavelength, flux_erg_aa)
    # plt.show()
    mock_gal_df = pd.DataFrame()


def generate_blackbody(T = 3500):
    bb = BlackBody(temperature=T*u.K)
    wav = np.arange(4000, 20000, 0.5) * u.AA
    flux = bb(wav)
    wavelength = wav.value # AA
    flux = 4*np.pi*flux.value # Jy
    wavelength_ha_idx = np.logical_and(wavelength > 6550, wavelength < 6580)
    wavelength_pab_idx = np.logical_and(wavelength > 13000, wavelength < 13050)
    scale_flux = np.max(flux[wavelength_ha_idx]) / 5e-7
    flux = flux / scale_flux
    return wavelength, flux

def add_emission_line(wavelength, flux_aa, peak, amplitude, sigma):
    gaussian_yvals = gaussian_func(wavelength, peak, amplitude, sigma)
    flux_aa = flux_aa + gaussian_yvals
    
    return flux_aa

def gaussian_func(wavelength, peak_wavelength, amp, sig):
    """Standard Gaussian funciton

    Parameters:
    wavelength (pd.DataFrame): Wavelength array to fit
    peak_wavelength (float): Peak of the line in the rest frame [angstroms]
    amp (float): Amplitude of the Gaussian
    sig (float): Standard deviation of the gaussian [angstroms]

    Returns:
    """
    return amp * np.exp(-(wavelength - peak_wavelength)**2 / (2 * sig**2))

generate_mock_lines()