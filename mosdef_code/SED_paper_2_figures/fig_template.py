import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import ascii
from plot_vals import *



def plot_name():
    fig, ax = plt.subplots(figsize = (8,8))

    fig.savefig(imd.cluster_paper_figures + 'fig_name.pdf', bbox_inches='tight')