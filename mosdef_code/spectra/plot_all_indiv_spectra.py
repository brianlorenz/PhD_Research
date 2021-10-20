# Plot all of the individual spectra for the galaxies in mosdef

import os
from mosdef_obj_data_funcs import get_mosdef_obj
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects
from read_data import mosdef_df
from astropy.io import fits
from spectra_funcs import read_axis_ratio_spectrum, clip_skylines, check_line_coverage, get_spectra_files, median_bin_spec, read_spectrum, get_too_low_gals, norm_spec_sed, read_composite_spectrum, prepare_mock_observe_spectrum, mock_observe_spectrum





fields_dict = {'ae': 'AEGIS', 'co': 'COSMOS', 'gn': 'GOODS-N', 'gs': 'GOODS-S', 'ud': 'UDS'}

def plot_all_spec():
    # Read in the axis ratio dataframe
    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()

    for i in range(len(ar_df)):
        field = ar_df.iloc[i]['field']
        v4id = ar_df.iloc[i]['v4id']
        mosdef_obj = get_mosdef_obj(field, v4id)

        # Find all the spectra files corresponding to this object
        spectra_files = get_spectra_files(mosdef_obj)
        for spectrum_file in spectra_files:
            spectrum_df = read_spectrum(mosdef_obj, spectrum_file)

            # Clip the skylines:
            spectrum_df['f_lambda_clip'], spectrum_df['mask'], spectrum_df['err_f_lambda_clip'] = clip_skylines(
                spectrum_df['obs_wavelength'], spectrum_df['f_lambda'], spectrum_df['err_f_lambda'], mask_negatives=False)
        
            # Make the figure
            fig, ax = plt.subplots(figsize=(8,8))
            ax.plot(spectrum_df['rest_wavelength'], spectrum_df['f_lambda'], color='red')
            ax.plot(spectrum_df['rest_wavelength'], spectrum_df['f_lambda_clip'], color='black')

            ax.set_ylim(np.min(spectrum_df['f_lambda_clip']), np.max(spectrum_df['f_lambda_clip']))

            fig.savefig(imd.mosdef_dir + f'/Spectra/1D_images/{field}_{v4id}_{spectrum_file}.pdf')

plot_all_spec()