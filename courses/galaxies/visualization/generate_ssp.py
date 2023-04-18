# Functions to generates an SSP and store it in the desired format
'''

'''


import fsps
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, SpanSelector
import time

os.environ["SPS_HOME"] = "/Users/brianlorenz/fsps/"

loc = '/Users/brianlorenz/code/courses/galaxies/visualization/'

ssp_folder = '/Users/brianlorenz/code/courses/galaxies/visualization/ssps/'

# High resolution for early times in star formation
high_res = 10**np.arange(-2, -0.1, 0.1)
# Low resolution for the rest of the galaxy
low_res = np.arange(0.5, 15.5, 0.5)
# All ages to create a spectrum from
ages_allowed = np.round(np.concatenate([high_res, low_res]), 3)


def get_ssp(metallicity=0.0, dust=0.0):
    """Generates a stellar population model with the given parameters

    Parameters:
    metallicity (float): log(Z) in solar units (so 0.0 is solar metallicity)
    dust (float): dust parameter

    Returns:
    Saves a series of files for a given stellar population over a range of ages from 0.01 to 15 in steps of 0.5

    """
    start_time = time.time()
    sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                                sfh=0, logzsol=metallicity, dust_type=2, dust2=dust, add_neb_emission=1)
    after_sp = time.time()
    print(f'sp generation: {after_sp - start_time}')
    sp.params['logzsol'] = metallicity
    sp.params['gas_logz'] = metallicity
    sp.params['gas_logu'] = -2.5
    for age in ages_allowed:
        sp_spectrum = sp.get_galaxy_spectrum(tage=age)[1]
        spec_time = time.time()
        print(f'spectrum_done: {spec_time - after_sp}')
        sp_stellar_mass = sp.stellar_mass
        sp_df = pd.DataFrame(sp_spectrum, columns=[
            'Spectrum'])
        sp_df['Stellar_mass'] = sp_stellar_mass
        filename = get_filename(metallicity=metallicity, dust=dust, age=age)
        sp_df.to_csv(ssp_folder+filename, index=False)
        if not os.path.exists(ssp_folder+'wavelength.df'):
            wavelength = sp.get_spectrum(tage=2)[0]
            wavelength_df = pd.DataFrame(wavelength, columns=['Wavelength'])
            wavelength_df.to_csv(ssp_folder+'wavelength.df', index=False)
    end_time = time.time()
    print(f'ages loop: {end_time - after_sp}')

def get_filename(metallicity=0.0, dust=0.0, age=1.0):
    """Generates the standard filename we use to access the files

    Parameters:
    metallicity (float): log(Z) in solar units (so 0.0 is solar metallicity)
    dust (float): dust parameter
    age (float): current age of stellar population in Gyr

    Returns:
    filename (str): Name of the file
    """
    filename = f'Z{metallicity}_d{dust}_t{age}.df'
    return filename

# get_ssp()
