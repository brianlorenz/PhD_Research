import pandas as pd
import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
import numpy as np
from axis_ratio_funcs import read_interp_axis_ratio, filter_ar_df, read_filtered_ar_df
from astropy.io import ascii
from axis_ratio_histogram import compare_ar_measurements

# Plot of F125 vs F160 fo rthe suspicious or removed gals

def make_removed_axis_gals_plot():

    ar_df = ascii.read(imd.mosdef_dir+'/axis_ratio_data/Merged_catalogs/filtered_ar_df_07_07.csv').to_pandas()
    ar_df_old = ascii.read(imd.mosdef_dir+'/axis_ratio_data/Merged_catalogs/filtered_ar_df_07_04.csv').to_pandas()

    # Finds galaxies that were in the old one but not the recent one
    removed_df = ar_df_old[~ar_df_old['v4id'].isin(ar_df['v4id'])]
    
    fig = compare_ar_measurements('F125_axis_ratio', 'F125_err_axis_ratio', 'F160_axis_ratio', 'F160_err_axis_ratio', save=False, ar_df_provided=removed_df)
    fig.savefig(imd.axis_output_dir + '/galfit_flags.pdf')
    


# make_removed_axis_gals_plot()