from astropy.io import ascii
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ssp_folder = '/Users/brianlorenz/code/courses/galaxies/visualization/ssps/'

def plot_ssp(metallicity, dust, age):
    wavelength_df = ascii.read(ssp_folder+'wavelength.df').to_pandas()

    filename = f'Z{metallicity}_d{dust}_t{age}.df'
    spec_df = ascii.read(ssp_folder+filename).to_pandas()
    spec_df['F_lambda_spectrum'] = wavelength_df['Wavelength']*spec_df['Spectrum']

    fig, ax = plt.subplots(figsize=(8,8))

    plot_range = np.logical_and(wavelength_df['Wavelength'] > 3000, wavelength_df['Wavelength'] < 10000)

    plt.plot(wavelength_df[plot_range]['Wavelength'], spec_df[plot_range]['F_lambda_spectrum'])
    plt.show()

def plot_two_ssp(metallicity1, dust1, age1, metallicity2, dust2, age2):
    wavelength_df = ascii.read(ssp_folder+'wavelength.df').to_pandas()

    def read_ssp_file(metallicity, dust, age):
        filename = f'Z{metallicity}_d{dust}_t{age}.df'
        spec_df = ascii.read(ssp_folder+filename).to_pandas()
        spec_df['F_lambda_spectrum'] = wavelength_df['Wavelength']*spec_df['Spectrum']
        return spec_df
    
    spec_df1 = read_ssp_file(metallicity1, dust1, age1)
    spec_df2 = read_ssp_file(metallicity2, dust2, age2)

    fig, ax = plt.subplots(figsize=(8,8))

    plot_range = np.logical_and(wavelength_df['Wavelength'] > 3000, wavelength_df['Wavelength'] < 10000)

    plt.plot(wavelength_df[plot_range]['Wavelength'], spec_df1[plot_range]['F_lambda_spectrum'], color='blue')
    plt.plot(wavelength_df[plot_range]['Wavelength'], spec_df2[plot_range]['F_lambda_spectrum'], color='orange')
    plt.show()

plot_ssp(0.0, 0.0, 1.0)
# plot_two_ssp(0.0, 0.0, 4.0, 0.0, 0.0, 4.0)
