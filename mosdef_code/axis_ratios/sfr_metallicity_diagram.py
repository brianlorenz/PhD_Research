# Plots the balmer decrement vs a vairiety of properies
from operator import methodcaller
from tkinter import font
import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
from matplotlib.patches import Ellipse
from ellipses_for_plotting import get_ellipse_shapes
import matplotlib as mpl
from plot_vals import *
from dust_model import *
from sympy.solvers import solve
from sympy import Symbol


def plot_sfr_metallicity_diagram():
    '''Generates a diagram of dust-to-gas ratio along sfr and metallicity'''
    
    fig = plt.figure(figsize=(8,8))

    #Add main axis
    ax = fig.add_axes([0.1, 0.08, 0.87, 0.87])
    plt.xticks([])
    plt.yticks([])

    # Labels and text
    fig.text(0.45, 0.02, 'SFR', fontsize=24)
    fig.text(0.3, 0.02, 'SFR', fontsize=24)
    fig.text(0.02, 0.45, 'Metallicity', fontsize=24, rotation=90)

    scale_aspect(ax)
    plt.show()


# plot_sfr_metallicity_diagram()