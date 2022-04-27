import initialize_mosdef_dirs as imd 
import matplotlib as mpl
from matplotlib.patches import Ellipse
from ellipses_for_plotting import get_ellipse_shapes
import matplotlib.pyplot as plt
import numpy as np
from axis_ratio_helpers import bootstrap_median
from sklearn.preprocessing import scale
from stack_spectra import norm_axis_stack
from matplotlib import patches
import matplotlib.gridspec as gridspec
from astropy.io import ascii
import pandas as pd
from sfms_bins import sfms_slope, sfms_yint

def simple_plot(n_clusters, save_name):

    fig, ax = plt.subplots(figsize=(8,8))

    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()

    ax.set_xlim(0, 1)
    ax.set_ylim(68, 110)

    for axis_group in range(n_clusters):
        row = summary_df.iloc[axis_group]
        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=-9.3, vmax=-8.1) 
        rgba = cmap(norm(row['log_use_ssfr_median']))

        ellipse_width, ellipse_height = get_ellipse_shapes(1, 42, row['shape'], scale_factor=0.025)

        emission_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits/{axis_group}_emission_fits.csv').to_pandas()
        ax.errorbar(row['use_ratio_median'], emission_df.iloc[0]['fixed_velocity'], yerr=emission_df.iloc[0]['err_fixed_velocity'], marker='None', ls='None', color=rgba)
        ax.add_artist(Ellipse((row['use_ratio_median'], emission_df.iloc[0]['fixed_velocity']), ellipse_width, ellipse_height, facecolor=rgba))
        ax.set_xlabel('Axis Ratio', fontsize=12)
        ax.set_ylabel('Velocity (km/s)', fontsize=12)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('log_ssfr', fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_aspect(ellipse_width/ellipse_height)
    
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/velocity_ar.pdf')
simple_plot(8, 'both_sfms_4bin_median_2axis_boot100')