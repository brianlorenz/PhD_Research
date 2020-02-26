# pypopstar_ps1.py
# Creates an isochrone using the pypopstar package

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
from popstar import synthetic, evolution, atmospheres, reddening, ifmr
from popstar.imf import imf, multiplicity


# Define isochrone parameters
logAges = [6.6, 7.6]  # Age in log(years)
AKs = 0  # extinction in mags
dist = 1000  # distance in parsec
metallicity = 0  # Metallicity in [M/H]

# Define evolution/atmosphere models and extinction law
evo_models = [evolution.MISTv1(), evolution.MergedSiessGenevaPadova()]
evo_model = evolution.MergedSiessGenevaPadova()
evo_model_name = ['MIST', 'Padova']
atm_func = atmospheres.get_merged_atmosphere
red_law = reddening.RedLawHosek18b()

# Also specify filters for synthetic photometry (optional). Here we use
# the HST WFC3-IR F127M, F139M, and F153M filters
filt_list = ['wfc3,ir,f127m', 'wfc3,ir,f139m', 'wfc3,ir,f153m']

# Make Isochrone object. Note that is calculation will take a few minutes, unless the
# isochrone has been generated previously.
for logAge in logAges:
    my_iso = synthetic.IsochronePhot(logAge, AKs, dist, metallicity=0,
                                     evo_model=evo_model, atm_func=atm_func,
                                     red_law=red_law, filters=filt_list)
    print(my_iso.save_file)
