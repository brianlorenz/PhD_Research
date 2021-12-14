# 6 panel figure with spectra overlaid in each group
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes

def plot_overlaid_spectra(savename):
    """Make the plot
    
    """
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{savename}/summary.csv').to_pandas()
    
    # Start the figure
    # fig, axarr = plt.subplots(3, 2, figsize=(14,10))
    fig = plt.figure(figsize=(10, 8))
    
    axarr = GridSpec(3,2, left=0.1, right=0.8, wspace=0.4, hspace=0.6)


    plot_lims = ((4850, 4875), (6540, 6590))

        
    bax_lowm_highs = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,0])
    bax_highm_highs = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,1])
    bax_lowm_lows = brokenaxes(xlims=plot_lims, subplot_spec=axarr[1,0])
    bax_highm_lows = brokenaxes(xlims=plot_lims, subplot_spec=axarr[1,1])
    bax_lowest_lows = brokenaxes(xlims=plot_lims, subplot_spec=axarr[2,0])
    bax_lowest_highs = brokenaxes(xlims=plot_lims, subplot_spec=axarr[2,1])


    for i in range(len(summary_df)):
        row = summary_df.iloc[i]
        axis_group = row['axis_group']

        if row['key'] == 'lowm_highs':
            ax = bax_lowm_highs
            ax.set_title('Low mass, high sSFR', color = row['color'])
        if row['key'] == 'highm_highs':
            ax = bax_highm_highs
            ax.set_title('High mass, high sSFR', color = row['color'])
        if row['key'] == 'lowm_lows':
            ax = bax_lowm_lows
            ax.set_title('Low mass, mid sSFR', color = row['color'])
        if row['key'] == 'highm_lows':
            ax = bax_highm_lows
            ax.set_title('High mass, mid sSFR', color = row['color'])
        if row['key'] == 'lowest_lows':
            ax = bax_lowest_lows
            ax.set_title('Low mass, low sSFR', color = row['color'])
        if row['key'] == 'lowest_highs':
            ax = bax_lowest_highs
            ax.set_title('High mass, low sSFR', color = row['color'])
        
        spec_df = ascii.read(imd.axis_cluster_data_dir + f'/{savename}/{savename}_spectra/{axis_group}_spectrum.csv').to_pandas()


        if row['shape'] == '+': 
            color = 'red'
            label = 'Axis Ratio < 0.4'
        if row['shape'] == 'd':
            color = 'mediumseagreen'
            label = '0.4 < Axis Ratio < 0.7'
        if row['shape'] == 'o':
            color = 'blue'
            label = '0.7 < Axis Ratio'

        # Find the peak of the halpha line so we can normalize it to 10^-17 erg/cm^2/s/anstrom
        halpha_range = np.logical_and(spec_df['wavelength']>6560, spec_df['wavelength']<6570)
        peak_halpha = np.max(spec_df[halpha_range]['f_lambda'])
        scale_factor = 1.0/peak_halpha
        ax.plot(spec_df['wavelength'], spec_df['f_lambda']*scale_factor, color=color, label = label) 
        ax.set_ylim(-0.1, 1.05)
        # ax.set_ylabel('F$_\\lambda$ ($10^{-17}$ erg/cm$^2$/s/$\AA$)')
        ax.set_ylabel('Normalized F$_\\lambda')
        ax.set_xlabel('Wavelength ($\AA$)')


        if i == len(summary_df)-1:
            ax.legend(bbox_to_anchor=(1.5, 4.5, 0.20, 0.15), loc='upper right')

        

    fig.savefig(imd.axis_cluster_data_dir + f'/{savename}/overlaid_spectra.pdf')


plot_overlaid_spectra('halpha_norm_oldsfr_newfit')