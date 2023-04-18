import fsps
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, SpanSelector
import time
from generate_ssp import get_filename
from astropy.io import ascii
import initialize_mosdef_dirs as imd


def make_ssp(metallicity, dust_value):
    sp = fsps.StellarPopulation(zcontinuous=1,
                                    add_neb_emission=1, dust_type=2, dust2=0)
    sp.params['logzsol'] = metallicity
    sp.params['gas_logz'] = metallicity
    sp.params['gas_logu'] = -2.5
    eline_waves = sp.emline_wavelengths
    eline_lums = sp.emline_luminosity
    wave, spec = sp.get_spectrum(tage=0.1, peraa=True)
    spec_df = pd.DataFrame(zip(wave, spec), columns=['wavelength', 'spectrum'])

    # filename = get_filename(metallicity=0.0, dust=0.0, age=1.0)

    # spec_df.to_csv('/Users/brianlorenz/code/courses/galaxies/visualization/test_ssps/' + filename)

    sp.params['dust2']= dust_value
    wave2, spec2 = sp.get_spectrum(tage=0.1, peraa=True)
    spec_df['spectrum_dusty'] = spec2

    spec_df.to_csv('dusty_spec.csv', index=False)

def smooth_sed(wavelength, spectrum, lower_wave, upper_wave):
    sed_point_size = 100
    new_waves = np.arange(lower_wave, upper_wave, sed_point_size)
    new_specs = []
    for new_wave in new_waves:
        target_waves = np.logical_and(wavelength > new_wave - sed_point_size/2, wavelength < new_wave + sed_point_size/2)
        new_spec = np.median(spectrum[target_waves])
        new_specs.append(new_spec)
    new_specs = np.array(new_specs)
    return new_waves, new_specs


def plot_fsps():
    spec_df = ascii.read('dusty_spec.csv').to_pandas()

    lower_wave = 2000
    upper_wave = 8000
    target_waves = np.logical_and(spec_df['wavelength'] > lower_wave, spec_df['wavelength'] < upper_wave)
    wave_idx = np.argmin(5500-spec_df['wavelength'])
    scale = spec_df.iloc[wave_idx]['spectrum'] / spec_df.iloc[wave_idx]['spectrum_dusty']
    # plt.plot(spec_df[target_waves]['wavelength'], spec_df[target_waves]['spectrum'])
    # plt.plot(spec_df[target_waves]['wavelength'], spec_df[target_waves]['spectrum_dusty'])
    dust_free_waves, dust_free_specs = smooth_sed(spec_df['wavelength'], spec_df['spectrum'], lower_wave, upper_wave)
    dusty_waves, dusty_spec = smooth_sed(spec_df['wavelength'], spec_df['spectrum_dusty'], lower_wave, upper_wave)
    
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.plot(dust_free_waves, dust_free_specs, marker='o', color='blue', ls='None', label='No Dust')
    ax.plot(dusty_waves, dusty_spec, marker='o', color='orange', ls='None', label='Dusty')
    ax.set_xlabel('Wavelength ($\AA$)', fontsize=16)
    ax.set_ylabel('Flux', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.set_yticks([]) 
    fig.savefig(imd.mosdef_dir + '/talk_plots/dusty_sed.pdf', bbox_inches='tight')


# make_ssp(-0.5, 0.8)
plot_fsps()