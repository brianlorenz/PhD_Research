# Makes a plot displaying all the clusters
from tokenize import group
import matplotlib.pyplot as plt
from composite_sed import vis_composite_sed
from astropy.io import ascii
import pandas as pd
import initialize_mosdef_dirs as imd
from plot_vals import *
from brokenaxes import brokenaxes
from sklearn import cluster
from spectra_funcs import read_composite_spectrum
from matplotlib.gridspec import GridSpec
from bpt_clusters_singledf import plot_bpt
from uvj_clusters import setup_uvj_plot
import sys

fontsize = 16

def make_overview_plot_clusters(n_clusters):
    #Set up the array (nrows, ncol)
    # fig, axarr = plt.subplots(n_clusters, 4, figsize=(16, n_clusters*4))

    fig = plt.figure(figsize=(17, n_clusters*4))
    gs = GridSpec(n_clusters, 5, left=0.05, right=0.95, wspace=0.3, hspace=0.6)

    clusters_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()

    filtered_gal_df = ascii.read(imd.loc_filtered_gal_df).to_pandas()
    filtered_gal_df['log_use_sfr'] = np.log10(filtered_gal_df['use_sfr'])

    clusters_summary_df_sorted=clusters_summary_df.sort_values('log_ssfr', ascending=False)
    #Manages which row to plot in
    plot_row_idx = 0

    for i in range(len(clusters_summary_df_sorted)): 
        if i >= n_clusters:
            print(f'Exiting after {n_clusters} rows, but there are {len(clusters_summary_df_sorted)} total')
            continue
        groupID = clusters_summary_df_sorted['groupID'].iloc[i]

        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        group_df['log_use_sfr'] = np.log10(group_df['use_sfr'])
        n_gals = len(group_df)
        # breakpoint()

        clusters_summary_row = clusters_summary_df[clusters_summary_df['groupID']==groupID]


        ### SED Plot --------------------------------------------
        # ax = axarr[groupID, 0]
        ax = fig.add_subplot(gs[plot_row_idx, 0])
        total_sed = ascii.read(imd.total_sed_csvs_dir + f'/{groupID}_total_sed.csv').to_pandas()
        
        vis_composite_sed(total_sed, composite_sed=0, composite_filters=0, groupID=groupID, std_scatter=0, run_filters=False, axis_obj=ax)
        
        ax.text(0.85, 0.85, f'{n_gals}', transform=ax.transAxes, fontsize=fontsize)
        ax.set_xlabel('Wavelength', fontsize=fontsize)
        ax.set_ylabel('Normalized Flux', fontsize=fontsize)
        ax.tick_params(labelsize = fontsize)
        # set_aspect_1(ax)





        ### Spectrum Plot --------------------------------------------
        # Set up the broken axis
        plot_lims = ((4850, 5020), (6535, 6595))
        ax = brokenaxes(xlims=plot_lims, subplot_spec=gs[plot_row_idx, 1])

        # Can change the norm_method here
        try:
            spec_df = read_composite_spectrum(groupID, 'cluster_norm', scaled='False')
            halpha_range = np.logical_and(spec_df['wavelength']>6560, spec_df['wavelength']<6570)
            peak_halpha = np.max(spec_df[halpha_range]['f_lambda'])
            scale_factor = 1.0/peak_halpha
            ax.plot(spec_df['wavelength'], spec_df['f_lambda']*scale_factor, color='black', linewidth=2)

            ax.set_xlabel('Wavelength', fontsize=fontsize, labelpad=30)
            ax.set_ylabel('Normalized Flux', fontsize=fontsize)
            ax.tick_params(labelsize = fontsize)
            ax.set_ylim(-0.1, 1.5)
        # set_aspect_1(ax)
        
        except:
            pass
        



        ### SFR/Mass Plot --------------------------------------------
        mass_lims = (7.9, 12.1)
        sfr_lims = (-0.5, 3)

        ax = fig.add_subplot(gs[plot_row_idx, 2])


        ax.plot(filtered_gal_df['log_mass'], filtered_gal_df['log_use_sfr'], marker='o', color='grey', ls='None', markersize=2)
        ax.plot(group_df['log_mass'], group_df['log_use_sfr'], marker='o', color='black', ls='None')
        ax.plot(clusters_summary_row['log_mass'], clusters_summary_row['log_sfr'], marker='x', color='red', ls='None', markersize=8, mew=2.5)

        # Count the galaxies
        in_mass_range = np.logical_and(group_df['log_mass']>mass_lims[0], group_df['log_mass']<mass_lims[1])
        in_sfr_range = np.logical_and(group_df['log_use_sfr']>sfr_lims[0], group_df['log_use_sfr']<sfr_lims[1])
        in_both_range = np.logical_and(in_mass_range, in_sfr_range)
        n_gals_in_range = len(group_df[in_both_range])
        ax.text(0.05, 0.85, f'{n_gals_in_range}', transform=ax.transAxes, fontsize=fontsize)

        # Plot lines of constant ssfr
        ssfrs = [0.1, 1, 10]
        ssfr_l_masses = np.arange(8, 13, 1)
        label_locs = [11.7, 11.7, 10.8]
        for i in range(len(ssfrs)):
            ssfr_l_sfrs = np.log10(10**ssfr_l_masses * ssfrs[i] / 10**9)
            label_loc = np.log10(10**label_locs[i] * ssfrs[i] / 10**9) - 0.1
            ax.plot(ssfr_l_masses, ssfr_l_sfrs, ls='--', color='orange')
            ax.text(label_locs[i], label_loc, f'{ssfrs[i]} Gyr$^{-1}$', rotation=50)

        ax.set_xlabel(stellar_mass_label, fontsize=fontsize)
        ax.set_ylabel(sfr_label, fontsize=fontsize)
        ax.tick_params(labelsize = fontsize)
        ax.set_xlim(mass_lims)
        ax.set_ylim(sfr_lims)
        # set_aspect_1(ax)



        ### UVJ Plot --------------------------------------------
        ax = fig.add_subplot(gs[plot_row_idx, 3])
        xrange = (-0.5, 2)
        yrange = (0, 2.5)

        # median_vj = np.median(group_df[group_df['V_J']>0]['V_J'])
        # median_uv = np.median(group_df[group_df['U_V']>0]['U_V'])

        setup_uvj_plot(ax, filtered_gal_df, 0, axis_obj=ax)
        ax.plot(group_df['V_J'], group_df['U_V'], marker='o', color='black', ls='None')
        ax.plot(clusters_summary_row['median_V_J'], clusters_summary_row['median_U_V'], marker='x', color='red', ls='None', markersize=8, mew=2.5)

        in_x_range = np.logical_and(group_df['V_J']>xrange[0], group_df['V_J']<xrange[1])
        in_y_range = np.logical_and(group_df['U_V']>yrange[0], group_df['U_V']<yrange[1])
        in_both_range_uvj = np.logical_and(in_x_range, in_y_range)
        n_gals_in_range_uvj = len(group_df[in_both_range_uvj])
        ax.text(0.85, 0.85, f'{n_gals_in_range_uvj}', transform=ax.transAxes, fontsize=fontsize)
        # ax.plot(group_df['log_mass'], group_df['log_use_sfr'], marker='o', color='black', ls='None')

        ax.set_xlabel('V-J', fontsize=fontsize)
        ax.set_ylabel('U-V)', fontsize=fontsize, labelpad=-7)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.tick_params(labelsize = fontsize)
        # set_aspect_1(ax)




        ### BPT Plot --------------------------------------------
        ax = fig.add_subplot(gs[plot_row_idx, 4])
        xrange = (-2, 1)
        yrange = (-1.2, 1.5)

        group_df_bpt = plot_bpt(axis_obj=ax, use_other_df=1, use_df=group_df, add_background=True)

        in_x_range = np.logical_and(group_df_bpt['log_NII_Ha']>xrange[0], group_df_bpt['log_NII_Ha']<xrange[1])
        in_y_range = np.logical_and(group_df_bpt['log_OIII_Hb']>yrange[0], group_df_bpt['log_OIII_Hb']<yrange[1])
        in_both_range_bpt = np.logical_and(in_x_range, in_y_range)
        n_gals_in_range_bpt = len(group_df[in_both_range_bpt])
        ax.text(0.85, 0.85, f'{n_gals_in_range_bpt}', transform=ax.transAxes, fontsize=fontsize)
        # ax.plot(group_df['log_mass'], group_df['log_use_sfr'], marker='o', color='black', ls='None')

        # Add the median of the cluster
        log_N2_Ha_group = clusters_summary_df_sorted['log_N2_Ha'].iloc[i]
        log_O3_Hb_group = clusters_summary_df_sorted['log_O3_Hb'].iloc[i]
        log_N2_Ha_group_errs = (clusters_summary_df_sorted['err_log_N2_Ha_low'].iloc[i], clusters_summary_df_sorted['err_log_N2_Ha_high'].iloc[i])
        log_O3_Hb_group_errs = (clusters_summary_df_sorted['err_log_O3_Hb_low'].iloc[i], clusters_summary_df_sorted['err_log_O3_Hb_high'].iloc[i])
        ax.plot(log_N2_Ha_group, log_O3_Hb_group, marker='x', color='red', markersize=10, mew=3, ls='None', zorder=10000)
        # ax.errorbar(log_N2_Ha_group, log_O3_Hb_group, xerr=log_N2_Ha_group_errs, yerr=log_O3_Hb_group_errs, marker='x', color='red', markersize=6, mew=3, ls='None')

        ax.set_xlabel('log(N[II]/H$\\alpha$)', fontsize=fontsize)
        ax.set_ylabel('log(O[III]/H$\\beta$)', fontsize=fontsize, labelpad=-7)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.tick_params(labelsize = fontsize)
        # set_aspect_1(ax)

        ax.text(1.1, 0.3, f'Group {groupID}', transform=ax.transAxes, fontsize=20, rotation=270)

        plot_row_idx = plot_row_idx + 1
    # plt.tight_layout()
    fig.savefig(imd.cluster_dir + '/cluster_stats/overview_clusters.pdf')


make_overview_plot_clusters(23)