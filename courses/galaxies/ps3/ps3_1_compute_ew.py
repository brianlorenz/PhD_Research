# Compute the equivalent widths for Mgb and Hb as per problem 1b/c

import sys
import os
import string
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
import pandas as pd

loc = '/Users/galaxies-air/Desktop/Galaxies/ps3/'

spectra_df = pd.read_csv(loc+'ssp_spectra.df')
spec_df_all = pd.read_csv(loc+'ssp_spectra_all.df')
ssp_3burst_df = pd.read_csv(loc+'ssp_3burst.df')


def find_EW(wavelength, spectrum, band, blue_cont, red_cont):
    '''
    Given a spectrum, band, and continuum range, comput the equivalent width for a line

    wavelength - entire wavelength of your spectrum
    spectrum - matches wavelength in length, contains flux values
    band - tuple - region over which to calculate EW
    blue_cont - tuple - region to compute continuum level
    red_cont - tuple - region to compute continuum level
    '''

    # indicies[0] for band
    # indicies[1] for blue
    # indicies[2] for red
    indicies = [np.where(np.logical_and(wavelength >= wave_band[0], wavelength <= wave_band[1]))
                for wave_band in (band, blue_cont, red_cont)]

    # Compute the continuum level given two band indices and a spectrum
    blue_mean = np.mean(spectrum[indicies[1][0]])
    red_mean = np.mean(spectrum[indicies[2][0]])
    continuum = np.mean([blue_mean, red_mean])

    # Old formula, wrong
    # Need to multiply by the separation in wavelength
    # scale = wavelength[indicies[0][0][1]] - wavelength[indicies[0][0][0]]
    # print(scale)

    # At each pixel, we want to sum the difference between that pixel and the continuum (wrong)
    # return scale*np.sum(continuum - spectrum[indicies[0][0]])
    return np.abs(np.sum(1-spectrum[indicies[0][0]]/continuum))


Mgb_band = (5160.125, 5192.625)
Mgb_blue = (5142.625, 5161.375)
Mgb_red = (5191.375, 5206.375)

Hb_band = (4847.875, 4876.625)
Hb_blue = (4827.875, 4847.875)
Hb_red = (4876.625, 4891.625)

ages = ('1', '2', '5', '14')
z_keys = ('low', 'mid', 'high')

keys = ['z'+z+'_t'+age for z in z_keys for age in ages]

ew_df = pd.DataFrame()
ew_df_partc = pd.DataFrame()

for key in keys:
    ew = find_EW(spectra_df['wavelength'], spectra_df[key],
                 Mgb_band, Mgb_blue, Mgb_red)
    ew_df[key+'_Mgb'] = [ew]

    ew = find_EW(spectra_df['wavelength'], spectra_df[key],
                 Hb_band, Hb_blue, Hb_red)
    ew_df[key+'_Hb'] = [ew]

for i in np.arange(0, 15):
    # Measures z0 from 0 to 14
    ew = find_EW(spectra_df['wavelength'], spec_df_all['zmid_t'+str(i)],
                 Mgb_band, Mgb_blue, Mgb_red)
    ew_df_partc['z0_t'+str(i)+'_Mgb'] = [ew]

    # Measures the combined ssp from 0 to 14
    ew = find_EW(spectra_df['wavelength'], ssp_3burst_df['t'+str(i)],
                 Mgb_band, Mgb_blue, Mgb_red)
    ew_df_partc['ssp_3burst_t'+str(i)+'_Mgb'] = [ew]

ew_df_partc.to_csv(loc + 'ew_partc.df')
