import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from tabulate import tabulate
from astropy.table import Table
import matplotlib.pyplot as plt


def plot_poly():
    """Given a field and id, read in the sed and create a plot of it

    Parameters:


    Returns:
    """
    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.tick_params(labelsize=ticksize, size=ticks)

    plt.show()
