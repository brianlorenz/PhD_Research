import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
import sys



def plot_emission_paper(n_bins, save_name):
    for axis_group in range(n_bins):
        cont_sub_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs/{axis_group}_cont_sub.csv').to_pandas()
        gauss_fit_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_gaussian_fits/{axis_group}_gaussian_fit.csv').to_pandas()
        print(cont_sub_df)

        fig, axarr = plt.subplots(1, 2, figsize=(8,8))
        ax_hb = axarr[0]
        ax_ha = axarr[1]
        hb_range = cont_sub_df['wavelength_cut'] < 6000
        
        def plot_on_axis(ax, range):
            """Keeps the style consistent between the axes
            
            Parameters:
            ax (matplotlib axis): axis to plot on 
            range (boolean array): How to filter the data           
            """
            ax.plot(cont_sub_df[range]['wavelength_cut'], cont_sub_df[range]['continuum_sub_ydata'], color='black')
            ax.plot(gauss_fit_df[range]['rest_wavelength'], gauss_fit_df[range]['gaussian_fit'], color='orange')
            
            ax.set_xlabel('Wavelength ($\AA$)')
            ax.set_ylabel('Flux')


        plot_on_axis(ax_hb, hb_range)
        plot_on_axis(ax_ha, ~hb_range)
        
        fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_images/{axis_group}_zoomed_out.pdf')
        plt.close()
plot_emission_paper(8, 'both_sfms_4bin_median_2axis_boot100')