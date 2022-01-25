import numpy as np
from numpy.core.defchararray import count
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
from axis_ratio_funcs import read_interp_axis_ratio, filter_ar_df
from save_counts import save_count




def count_gals_by_step():
    """Runs counts of all the various steps in the data processing pipeline
    """
    # Axis ratio interpolation
    ar_df = read_interp_axis_ratio()
    # Check if the ratio used is not an interpolation, but a match
    f125_match = ar_df['use_ratio'] == ar_df['F125_axis_ratio']
    f160_match = ar_df['use_ratio'] == ar_df['F160_axis_ratio']
    match = np.logical_or(f125_match, f160_match)
    ar_df_match = ar_df[match]
    save_count(ar_df_match, 'axis_ratio_one_bad', 'Galaxies not using interpolation')

    # All counts are embedded into the filtering to get a step-by-step count
    filter_ar_df(ar_df)


count_gals_by_step()