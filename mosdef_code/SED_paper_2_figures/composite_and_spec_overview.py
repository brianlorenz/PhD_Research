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
from fit_emission import line_list

single_column_axisfont = 14



def composite_and_spec_overview(n_clusters, ignore_groups):
    clusters_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    clusters_summary_df = clusters_summary_df[~clusters_summary_df['groupID'].isin(ignore_groups)]

    filtered_gal_df = ascii.read(imd.loc_filtered_gal_df).to_pandas()
    filtered_gal_df['log_use_sfr'] = np.log10(filtered_gal_df['use_sfr'])

    fig = plt.figure(figsize=(12, len(clusters_summary_df)*2))
    nrows = math.ceil(len(clusters_summary_df)/2)
    gs = GridSpec(nrows, 5, left=0.05, right=0.95, wspace=0.4, hspace=1.0, width_ratios=[1, 1, 0.4, 1, 1])

    clusters_summary_df_sorted=clusters_summary_df.sort_values('computed_log_ssfr', ascending=False)
    
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
        plot_lims = ((4850, 5020), (6530, 6595))
        ax_spec = brokenaxes(xlims=plot_lims, subplot_spec=gs[i%nrows, plot_col_index+1])

        spec_df = read_composite_spectrum(groupID, 'cluster_norm', scaled='False')
        halpha_range = np.logical_and(spec_df['wavelength']>6560, spec_df['wavelength']<6570)
        peak_halpha = np.max(spec_df[halpha_range]['f_lambda'])
        scale_factor = 1.0/peak_halpha
        ax_spec.plot(spec_df['wavelength'], spec_df['f_lambda']*scale_factor, color='black', linewidth=2, label='Composite')

        ax_spec.set_ylim(-0.1, 1.5)
        ax_spec.set_xlabel('Rest Wavelength ($\AA$)', fontsize=single_column_axisfont, labelpad=30)
        ax_spec.set_ylabel('Normalized Flux', fontsize=single_column_axisfont, labelpad=16)
        ax_spec.tick_params(labelsize = single_column_axisfont)

        # Label the emisison lines
        for line in line_list:
            name = line[0]
            center = line[1]
            # line_range = np.logical_and(spec_df['wavelength']>(center-5), spec_df['wavelength']<(center+5))
            line_range = np.logical_and(spec_df['wavelength']>(center-1), spec_df['wavelength']<(center+1))
            height = np.max(spec_df[line_range]['f_lambda']*scale_factor)
            ylims = ax_spec.get_ylim()[0]
            height_pct = (height-ylims[0]) / (ylims[1]-ylims[0])
            # ax.axvline(center, ymin=-0, ymax=0.78, color='black', ls='--')
            ax_spec.axvline(center, color='mediumseagreen', ls='--')
            if len(name) > 8:
                offset = -8
            else:
                offset = len(name)*-2.7
            if i%nrows==0:
                if name=='Halpha': name='H$\\alpha$'
                if name=='Hbeta': name='H$\\beta$'
                if name=='O3_5008' or name=='O3_4960': name='O[III]'
                if name=='N2_6550' or name=='N2_6585': name='N[II]'
                ax_spec.text(center+3+offset, np.min([ylims[1]-0.1, height+0.2]), name, fontsize=single_column_axisfont)

        
        

    fig.savefig(imd.cluster_paper_figures + '/composite_and_spec_overview.pdf', bbox_inches='tight')

ignore_groups = imd.ignore_groups
composite_and_spec_overview(23, ignore_groups)