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

def plot_overlaid_spectra(savename, plot_cont_sub=False, paper_fig=False):
    """Make the plot

    Parameters:
    savename (str): Folder to save the name under
    plot_cont_sub (boolean): Set to True to plot continuum-subtracted data
    paper_fig (boolean): Set to trun to use alternate settings to gneerate a figure for publication - better titles and labels, etc
    """
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{savename}/summary.csv').to_pandas()
    
    # Start the figure
    # fig, axarr = plt.subplots(3, 2, figsize=(14,10))
    fig = plt.figure(figsize=(10, 8))
    
    n_rows = int(len(summary_df) / 6)
    if savename=='both_sfms_6bin_median_2axis':
        n_rows=3
    if savename=='both_sfms_6bin_median_1axis':
        n_rows=3
    if savename=='both_6bin_1axis_median_params':
        n_rows=3
    if len(summary_df) == 8:
        n_rows = 2
    if len(summary_df) == 4:
        n_rows = 2
    axarr = GridSpec(n_rows, 2, left=0.1, right=0.8, wspace=0.4, hspace=0.6)


    plot_lims = ((4850, 4875), (6540, 6590))

    if n_rows == 1:
        bax_0 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,0])
        bax_1 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,1])
    if n_rows > 1:
        bax_0 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[1,0])
        bax_1 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,0])
        bax_2 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[1,1])
        bax_3 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,1])
    if n_rows > 2:
        bax_0 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[2,0])
        bax_1 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[1,0])
        bax_2 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,0])
        bax_3 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[2,1])
        bax_4 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[1,1])
        bax_5 = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,1])


    for i in range(len(summary_df)):
        row = summary_df.iloc[i]
        axis_group = row['axis_group']

        if row['key'] == 'sorted0':
            ax = bax_0
            ax.set_title('Sorted 0', color = row['color'])
            if paper_fig == True:
                 ax.set_title('$\log(M_*) < 10$, below SFMS', color = row['color'])
        if row['key'] == 'sorted1':
            ax = bax_1
            ax.set_title('Sorted 1', color = row['color'])
            if paper_fig == True:
                 ax.set_title('$\log(M_*) < 10$, above SFMS', color = row['color'])
        if row['key'] == 'sorted2':
            ax = bax_2
            ax.set_title('Sorted 2', color = row['color'])
            if paper_fig == True:
                 ax.set_title('$\log(M_*) > 10$, below SFMS', color = row['color'])
        if row['key'] == 'sorted3':
            ax = bax_3
            ax.set_title('Sorted 3', color = row['color'])
            if paper_fig == True:
                 ax.set_title('$\log(M_*) > 10$, above SFMS', color = row['color'])
        if row['key'] == 'sorted4':
            ax = bax_4
            ax.set_title('Sorted 4', color = row['color'])
        if row['key'] == 'sorted5':
            ax = bax_5
            ax.set_title('Sorted 5', color = row['color'])
        
        ### Read in spectra
        if plot_cont_sub==True:
            spec_df = ascii.read(imd.axis_cluster_data_dir + f'/{savename}/{savename}_cont_subs/{axis_group}_cont_sub.csv').to_pandas()
            add_str = ' (Cont-sub)'
            spec_df = spec_df.rename(columns={"wavelength_cut": "wavelength", "continuum_sub_ydata": "f_lambda"})
        else:
            add_str = ''
            spec_df = ascii.read(imd.axis_cluster_data_dir + f'/{savename}/{savename}_spectra/{axis_group}_spectrum.csv').to_pandas()
  
        if row['shape'] == '+': 
            color = 'red'
            label = 'Axis Ratio < 0.4'
            if paper_fig==True:
                label = '$(b/a) < 0.55$'
        if row['shape'] == 'd':
            color = 'mediumseagreen'
            label = '0.4 < Axis Ratio < 0.7'
        if row['shape'] == 'o':
            color = 'blue'
            label = '0.7 < Axis Ratio'
            if paper_fig==True:
                label = '$(b/a) > 0.55$'
        if row['shape'] == 1.0: 
            color = 'red'
            label = ''



        # Find the peak of the halpha line so we can normalize it to 10^-17 erg/cm^2/s/anstrom
        halpha_range = np.logical_and(spec_df['wavelength']>6560, spec_df['wavelength']<6570)
        peak_halpha = np.max(spec_df[halpha_range]['f_lambda'])
        scale_factor = 1.0/peak_halpha
        ax.plot(spec_df['wavelength'], spec_df['f_lambda']*scale_factor, color=color, label = label) 
        ax.set_ylim(-0.1, 1.05)
        ax.set_ylabel(f'Normalized F$_\\lambda${add_str}')
        ax.set_xlabel('Wavelength ($\AA$)')
        if paper_fig == True:
            ax.set_ylabel(f'Normalized Flux')
            ax.set_xlabel('Wavelength ($\AA$)')

        if paper_fig==True:
            if i == 5:
                ax.legend(bbox_to_anchor=(0.36, 0.85, 0.20, 0.15), loc='upper right')

        
    if paper_fig==True:
        fig.savefig(imd.axis_cluster_data_dir + f'/{savename}/overlaid_spectra_paper.pdf')
    else:
        fig.savefig(imd.axis_cluster_data_dir + f'/{savename}/overlaid_spectra.pdf')
    plt.close('all')

# plot_overlaid_spectra('mosdef_ssfr_4bin_mean', plot_cont_sub=True)
plot_overlaid_spectra('both_sfms_4bin_median_2axis_boot100', plot_cont_sub=True, paper_fig=True)