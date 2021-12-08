# Plots the theoretical distribution of axis rations

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed
from filter_response import lines, overview, get_index, get_filter_response
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from query_funcs import get_zobjs, get_zobjs_sort_nodup
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf






def compute_theory(q, q0=0.5):
    '''f
    
    Parameters:
    q (float): Variable, overved axis ratio
    q0 (float): Intrinsic axis ratio of the distribution, assume all galaxies have this
    '''

    # def func_to_integrate(q):
    #     integrand = q0 / ((1-q0**2)*((q**2-q0**2)**(1/2)))
    #     return integrand

    # print(func_to_integrate(0.6))

    # f_q = q * integrate.quad(func_to_integrate, 0.50, q)[0]
    f_q = q * (q0 / ((1-q0**2)*((q**2-q0**2)**(1/2))))
    return f_q


def make_plot(q0=0.7):
    qs = np.arange(q0+0.01, 1, 0.01)
    yvals = [compute_theory(q, q0=q0) for q in qs]
    plt.plot(qs, yvals)
    plt.show()


make_plot()