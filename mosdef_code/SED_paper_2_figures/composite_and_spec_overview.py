import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import ascii
from plot_vals import *
from brokenaxes import brokenaxes
from spectra_funcs import read_composite_spectrum
from matplotlib.gridspec import GridSpec
import math
from composite_sed import vis_composite_sed

single_column_axisfont = 14


def composite_and_spec_overview(n_clusters):
    fig = plt.figure(figsize=(12, 40))
    nrows = math.ceil(n_clusters/2)
    gs = GridSpec(nrows, 5, left=0.05, right=0.95, wspace=0.4, hspace=1.0, width_ratios=[1, 1, 0.4, 1, 1])

    clusters_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()

    filtered_gal_df = ascii.read(imd.loc_filtered_gal_df).to_pandas()
    filtered_gal_df['log_use_sfr'] = np.log10(filtered_gal_df['use_sfr'])

    clusters_summary_df_sorted=clusters_summary_df.sort_values('median_log_ssfr', ascending=False)
    
    for i in range(len(clusters_summary_df_sorted)):
        groupID = clusters_summary_df_sorted['groupID'].iloc[i]

        if i >= nrows:
            plot_col_index = 3
        else:
            plot_col_index = 0
        ax_composite = fig.add_subplot(gs[i%nrows, plot_col_index])

        ### SED Plot --------------------------------------------
        total_sed = ascii.read(imd.total_sed_csvs_dir + f'/{groupID}_total_sed.csv').to_pandas()
        
       

        vis_composite_sed(total_sed, composite_sed=0, composite_filters=0, groupID=groupID, std_scatter=0, run_filters=False, axis_obj=ax_composite, grey_points=True, errorbars=False)
        
        ax_composite.set_xlabel('Rest Wavelength ($\AA$)', fontsize=single_column_axisfont)
        ax_composite.set_ylabel('Flux', fontsize=single_column_axisfont)
        ax_composite.tick_params(labelsize = single_column_axisfont)
        # ax_composite.text(0.85, 0.85, f'{groupID}', transform=ax_composite.transAxes, fontsize=single_column_axisfont)
        

        ### Spectrum Plot --------------------------------------------
        # Set up the broken axis
        plot_lims = ((4850, 5020), (6535, 6595))
        ax_spec = brokenaxes(xlims=plot_lims, subplot_spec=gs[i%nrows, plot_col_index+1])

        # Can change the norm_method here
        try:
            spec_df = read_composite_spectrum(groupID, 'cluster_norm', scaled='False')
            halpha_range = np.logical_and(spec_df['wavelength']>6560, spec_df['wavelength']<6570)
            peak_halpha = np.max(spec_df[halpha_range]['f_lambda'])
            scale_factor = 1.0/peak_halpha
            ax_spec.plot(spec_df['wavelength'], spec_df['f_lambda']*scale_factor, color='black', linewidth=2, label='Composite')

            ax_spec.set_ylim(-0.1, 1.5)
            ax_spec.set_xlabel('Rest Wavelength ($\AA$)', fontsize=single_column_axisfont, labelpad=30)
            ax_spec.set_ylabel('Normalized Flux', fontsize=single_column_axisfont, labelpad=16)
            ax_spec.tick_params(labelsize = single_column_axisfont)
            
        except:
            pass


    fig.savefig(imd.cluster_paper_figures + '/composite_and_spec_overview.pdf', bbox_inches='tight')

composite_and_spec_overview(23)