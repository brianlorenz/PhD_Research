import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from tabulate import tabulate
from astropy.table import Table
from read_data import mosdef_df
import matplotlib.pyplot as plt


def plot_sed(sed):
    """Given an sed dataframe with a wavelength and magnitude column, create a plot

    Parameters:
    sed (pd.DataFrame) = dataframe containing the columns 'magnitude' and 'wavelength'


    Returns:
    """
    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.errorbar(sed['peak_wavelength'], sed['magnitude'], yerr=[
        sed['magnitude_err_lower'], sed['magnitude_err_upper']], ls='None', color='black', marker='o')
    # ax.errorbar(sed['peak_wavelength'], sed['flux'], yerr=[sed['flux'] - sed['flux_error'],
    #                                                       sed['flux'] + sed['flux_error']], ls='None', color='black', marker='o')
    # plt.yscale('log')
    #ax.set_ylim(-10, 10)
    ax.invert_yaxis()
    ax.set_xlabel('Wavelength ($\AA$)', fontsize=axisfont)
    ax.set_ylabel('Magnitude', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    plt.show()
