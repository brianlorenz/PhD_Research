# Makes a plot displaying all the clusters
from tokenize import group
import matplotlib.pyplot as plt
from composite_sed import vis_composite_sed
from astropy.io import ascii
import pandas as pd
import initialize_mosdef_dirs as imd
from plot_vals import *
from brokenaxes import brokenaxes
from spectra_funcs import read_composite_spectrum
from matplotlib.gridspec import GridSpec
from bpt_clusters_singledf import plot_bpt
from uvj_clusters import setup_uvj_plot
import matplotlib as mpl
from composite_and_spec_overview import label_elines
import pickle 



prospector_run = 'dust_index_test'

fontsize = 24
rows_per_page = 7


def setup_figs(n_clusters, norm_method, color_gals=False, bpt_color=False, paper_overview=False, prospector_spec=True):
    # Sort the groups
    clusters_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    clusters_summary_df_sorted = clusters_summary_df.sort_values('median_log_mass', ascending=True)
    groupIDs = clusters_summary_df_sorted['groupID'].to_numpy()
    #Manually order them here
    # groupIDs = [3, 12, 14, 2, 6, 10, 5, 13, 19, 16, 9, 17, 7, 1, 0, 11, 4, 8, 18, 15]
    groupID_dict = {}
    for i in range(len(groupIDs)):
        groupID_dict[groupIDs[i]] = i+1

    with open(imd.cluster_dir + '/groupID_dict.pkl', 'wb') as f:
        pickle.dump(groupID_dict, f)
        
    
    #Assign colors
    cmap = mpl.cm.coolwarm
    # norm = mpl.colors.Normalize(vmin=0.5, vmax=1.5) 
    # rgbas = [cmap(norm(clusters_summary_df[clusters_summary_df['groupID'] == i]['median_U_V'].iloc[0])) for i in groupIDs]
    norm = mpl.colors.Normalize(vmin=9.4, vmax=11.05) 
    rgbas = [cmap(norm(clusters_summary_df[clusters_summary_df['groupID'] == i]['median_log_mass'].iloc[0])) for i in groupIDs]
    color_df = pd.DataFrame(zip(groupIDs, rgbas), columns=['groupID', 'rgba'])
    color_df.to_csv(imd.loc_color_df, index=False)

    # Split into multiple pages if making the plot for the paper
    groupID_sets = []
    save_strs = []
    if paper_overview==True:
        n_figures = int(np.ceil(n_clusters / rows_per_page))
        for page_num in range(n_figures):
            # groupID_sets.append(groupIDs[(rows_per_page*page_num) : rows_per_page*(1+page_num)])
            save_strs.append(f'_page{page_num}')
        # Hard-coded for first page to be shorter
        groupID_sets.append(groupIDs[0 : 6])
        groupID_sets.append(groupIDs[6 : 13])
        groupID_sets.append(groupIDs[13 : 20])
    else:
        groupID_sets.append(groupIDs)
        save_strs.append('')
    for i in range(len(groupID_sets)):
        groupIDs = groupID_sets[i]
        save_str = save_strs[i]
        
        make_overview_plot_clusters(groupIDs, save_str, n_clusters, norm_method, groupID_dict, color_gals=color_gals, bpt_color=bpt_color, paper_overview=paper_overview, prospector_spec=prospector_spec)


def make_overview_plot_clusters(groupIDs, import_save_str, n_clusters, norm_method, groupID_dict, color_gals=False, bpt_color=False, paper_overview=False, prospector_spec=True):
    """
    Parameters:
    groupIDs (list): Which group IDs to plot, in order. Will also set the number of rows
    prospector_spec (boolean): Set to true (and update runname above) to add prospector spectra to plot
    
    """

    n_rows = len(groupIDs)
    if paper_overview == False:
        fig = plt.figure(figsize=(24, n_rows*4))
    else:
        fig = plt.figure(figsize=(24, n_rows*4))
    gs = GridSpec(n_rows, 5, left=0.02, right=0.98, wspace=0.28, hspace=0.07, top=0.98, bottom=0.02)

    
    clusters_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()

    filtered_gal_df = ascii.read(imd.loc_filtered_gal_df).to_pandas()
    filtered_gal_df['log_use_sfr'] = np.log10(filtered_gal_df['use_sfr'])

    similarity_df = ascii.read(imd.cluster_dir+'/cluster_stats/similarities/composite_similarities.csv').to_pandas()
    clusters_summary_df = clusters_summary_df.merge(similarity_df, on='groupID')
    
    #Manages which row to plot in
    plot_row_idx = 0
    add_xaxis_labels = [n_rows-1]
    add_yaxis_labels = range(0, n_rows)

    for groupID in groupIDs: 
        i = clusters_summary_df[clusters_summary_df['groupID'] == groupID].index[0]

        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        group_df['log_use_sfr'] = np.log10(group_df['use_sfr'])
        n_gals = len(group_df)

        clusters_summary_row = clusters_summary_df[clusters_summary_df['groupID']==groupID]

        if bpt_color==True:
            ax = fig.add_subplot(gs[plot_row_idx, 0])
            group_df_bpt = plot_bpt(axis_obj=ax, use_other_df=1, use_df=group_df, add_background=True, color_gals=color_gals)
            
            fig.delaxes(ax)

        ### SED Plot --------------------------------------------
        # ax = axarr[groupID, 0]
        ax = fig.add_subplot(gs[plot_row_idx, 0])
        total_sed = ascii.read(imd.total_sed_csvs_dir + f'/{groupID}_total_sed.csv').to_pandas()
        
        vis_composite_sed(total_sed, composite_sed=0, composite_filters=0, groupID=groupID, std_scatter=0, run_filters=False, axis_obj=ax, grey_points=True, errorbars=False, scale_5000=True)
        
        if paper_overview==False:
            ax.text(0.85, 0.85, f'{n_gals}', transform=ax.transAxes, fontsize=fontsize)
        if paper_overview==True:
            ax.text(0.97, 0.91, f'Group {groupID_dict[groupID]}', transform=ax.transAxes, fontsize=fontsize-2, horizontalalignment='right')
            ax.text(0.97, 0.83, f'S = {clusters_summary_row["mean_sim_to_composite"].iloc[0]:.2f}', transform=ax.transAxes, fontsize=fontsize-2, horizontalalignment='right')
            ax.text(0.97, 0.75, f'N = {n_gals}', transform=ax.transAxes, fontsize=fontsize-2, horizontalalignment='right')

        if plot_row_idx in add_xaxis_labels:
            ax.set_xlabel('Wavelength', fontsize=fontsize)
        else:
            ax.set_xlabel('')
        if plot_row_idx in add_yaxis_labels:
            ax.set_ylabel('F$_\lambda$ (norm)', fontsize=fontsize)
        else:
            ax.set_ylabel('')

        if groupID != groupIDs[-1]:
            ax.tick_params(labelbottom=False)   
        ax.tick_params(labelsize = fontsize)
        set_aspect_1(ax)





        ### Spectrum Plot --------------------------------------------
        # Set up the broken axis
        plot_lims = ((4850, 5020), (6520, 6605))
        ax = brokenaxes(xlims=plot_lims, subplot_spec=gs[plot_row_idx, 1])
        ax.set_ylim(-0.1, 1.55)

        # Can change the norm_method here
        spec_df = read_composite_spectrum(groupID, norm_method, scaled='False')
        halpha_range = np.logical_and(spec_df['wavelength']>6560, spec_df['wavelength']<6570)
        peak_halpha = np.max(spec_df[halpha_range]['f_lambda'])
        scale_factor = 1.0/peak_halpha
        ax.plot(spec_df['wavelength'], spec_df['f_lambda']*scale_factor, color='black', linewidth=2, label='Composite')

        if groupID == groupIDs[0]:
            label_elines(ax, spec_df, scale_factor)

        ax.axs[0].set_xticks([4900, 5000])
        # import matplotlib.ticker as ticker
        # ax.axs[1].xaxis.set_major_locator(ticker.FixedLocator((6550)))

        # ax.axs[1].set_xticks([6550])
        # ax.axs[1].set_xticks([6550])
        # ax.axs[1].minorticks_on()
        # ax.axs[1].set_xticks([6550], minor=True)
        ax.axs[0].minorticks_on()
        ax.axs[1].minorticks_on()
        # from matplotlib.ticker import FormatStrFormatter
        # ax.axs[1].xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))

        if plot_row_idx in add_xaxis_labels:
            ax.set_xlabel('Wavelength', fontsize=fontsize, labelpad=30)
        if plot_row_idx in add_yaxis_labels:
            ax.set_ylabel('F$_\lambda$ (norm)', fontsize=fontsize, labelpad=45)
        if groupID != groupIDs[-1]:
            ax.tick_params(labelbottom=False)   
        # ax.tick_params(labelsize = fontsize)
        ax.axs[0].tick_params(labelsize = fontsize)
        ax.axs[1].tick_params(labelsize = fontsize)
        ax.set_ylim(-0.1, 1.5)

        
        # scale_aspect(ax)

        # Add prospector spectrum
        if prospector_spec == True:
            try:
                prospector_spec_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{prospector_run}_csvs/group{groupID}_spec.csv').to_pandas()
                halpha_range = np.logical_and(prospector_spec_df['rest_wavelength']>6560, prospector_spec_df['rest_wavelength']<6570)
                peak_halpha = np.max(prospector_spec_df[halpha_range]['spec50_flambda'])
                scale_factor = 1.0/peak_halpha
                
                ax.plot(prospector_spec_df['rest_wavelength'], prospector_spec_df['spec50_flambda']*scale_factor, color='orange', linewidth=2, label='Prospector')
                # ax.legend()
            except:
                pass 



        ### SFR/Mass Plot --------------------------------------------
        mass_lims = (7.9, 12.1)
        sfr_lims = (-0.5, 3)

        ax = fig.add_subplot(gs[plot_row_idx, 2])

        ax.plot(filtered_gal_df['log_mass'], filtered_gal_df['log_use_sfr'], marker='o', color=grey_point_color, ls='None', markersize=grey_point_size)
        # ax.plot(clusters_summary_row['median_log_mass'], clusters_summary_row['median_log_sfr'], marker='x', color='red', ls='None', markersize=10, mew=3, zorder=10000)
        ax.plot(clusters_summary_row['median_log_mass'], clusters_summary_row['computed_log_sfr_with_limit'], marker=cluster_marker, color=get_row_color(groupID), mew=paper_marker_edge_width, mec=paper_mec, ls='None', markersize=get_row_size(i), zorder=10000)
        computed_sfr = clusters_summary_row['computed_log_sfr_with_limit']
        if clusters_summary_row['flag_balmer_lower_limit'].iloc[0] == 1:
            # ax.vlines(clusters_summary_row['median_log_mass'], computed_sfr, computed_sfr+100000, color=get_row_color(groupID))
            arrow_width = 0.01
            arrow_length = 0.06
            mass_xrange = 12.1-7.9
            sfr_xrange = 3-(-0.5)
            # breakpoint()
            ax.arrow((clusters_summary_row['median_log_mass'].iloc[0]-mass_lims[0])/mass_xrange, (clusters_summary_row['computed_log_sfr_with_limit'].iloc[0]-sfr_lims[0])/sfr_xrange, 0, arrow_length, color=get_row_color(groupID), width=arrow_width, transform=ax.transAxes)
        
        cmap = mpl.cm.plasma
        norm = mpl.colors.Normalize(vmin=1, vmax=n_gals) 
        for gal in range(len(group_df)):
            row = group_df.iloc[gal]
            if color_gals:
                rgba = cmap(norm(row['group_gal_id']))
            else:
                # cmap = mpl.cm.plasma
                # norm = mpl.colors.Normalize(vmin=-0.5, vmax=1.0) 
                # o3hb = np.log10(row['oiii_5008_flux']/row['hb_flux'])
                # rgba = cmap(norm(o3hb))
                rgba = 'black'
        
            ax.plot(row['log_mass'], row['log_use_sfr'], marker='o', color=rgba, ls='None')



        # Count the galaxies
        in_mass_range = np.logical_and(group_df['log_mass']>mass_lims[0], group_df['log_mass']<mass_lims[1])
        in_sfr_range = np.logical_and(group_df['log_use_sfr']>sfr_lims[0], group_df['log_use_sfr']<sfr_lims[1])
        in_both_range = np.logical_and(in_mass_range, in_sfr_range)
        has_hbeta_too = np.logical_and(in_both_range, group_df['hb_detflag_sfr']==0)
        n_gals_in_range = len(group_df[in_both_range])
        n_gals_hbeta = len(group_df[has_hbeta_too])
        ax.text(0.03, 0.90, f'{n_gals_in_range} H$\\alpha$', transform=ax.transAxes, fontsize=fontsize)
        ax.text(0.03, 0.80, f'{n_gals_hbeta} H$\\beta$', transform=ax.transAxes, fontsize=fontsize)
        # if paper_overview==False:
        #     ax.text(0.05, 0.85, f'{n_gals_in_range}', transform=ax.transAxes, fontsize=fontsize)

        # Plot lines of constant ssfr
        ssfrs = [0.1, 1, 10]
        ssfr_l_masses = np.arange(8, 13, 1)
        label_locs = [11.7, 11.7, 10.8]
        for k in range(len(ssfrs)):
            ssfr_l_sfrs = np.log10(10**ssfr_l_masses * ssfrs[k] / 10**9)
            label_loc = np.log10(10**label_locs[k] * ssfrs[k] / 10**9) - 0.1
            ax.plot(ssfr_l_masses, ssfr_l_sfrs, ls='--', color='orange')
            if groupID == groupIDs[0]:
                ax.text(label_locs[k], label_loc, f'{ssfrs[k]} Gyr$^{-1}$', rotation=50, fontsize=12)

        if plot_row_idx in add_xaxis_labels:
            ax.set_xlabel(stellar_mass_label, fontsize=fontsize)
        if plot_row_idx in add_yaxis_labels:
            ax.set_ylabel(sfr_label, fontsize=fontsize, labelpad=2)
        if groupID != groupIDs[-1]:
            ax.tick_params(labelbottom=False)   
        ax.tick_params(labelsize = fontsize)
        ax.set_xlim(mass_lims)
        ax.set_ylim(sfr_lims)
        scale_aspect(ax)



        ### UVJ Plot --------------------------------------------
        ax = fig.add_subplot(gs[plot_row_idx, 3])
        xrange = (-0.5, 2)
        yrange = (0, 2.5)

        # median_vj = np.median(group_df[group_df['V_J']>0]['V_J'])
        # median_uv = np.median(group_df[group_df['U_V']>0]['U_V'])

        setup_uvj_plot(ax, filtered_gal_df, 0, axis_obj=ax)
        cmap = mpl.cm.plasma
        norm = mpl.colors.Normalize(vmin=1, vmax=n_gals) 
        print(n_gals)
        for gal in range(len(group_df)):
            row = group_df.iloc[gal]
            if color_gals:
                rgba = cmap(norm(row['group_gal_id']))
            else:
                rgba = 'black'
        
            ax.plot(row['V_J'], row['U_V'], marker='o', color=rgba, ls='None')

        # Add the median
        # ax.plot(clusters_summary_row['median_V_J'], clusters_summary_row['median_U_V'], marker='x', color='red', ls='None', markersize=10, mew=3)
        # Add the measurement from the composite uvj
        composite_uvj_df = ascii.read(imd.composite_uvj_dir + '/composite_uvjs.csv').to_pandas()
        uvj_composite = composite_uvj_df[composite_uvj_df['groupID'] == groupID]
        ax.plot(uvj_composite['V_J'], uvj_composite['U_V'],
            ls='', marker=cluster_marker, markersize=paper_marker_size, color=get_row_color(groupID), mew=paper_marker_edge_width, mec=paper_mec, label='Composite SED')

        in_x_range = np.logical_and(group_df['V_J']>xrange[0], group_df['V_J']<xrange[1])
        in_y_range = np.logical_and(group_df['U_V']>yrange[0], group_df['U_V']<yrange[1])
        in_both_range_uvj = np.logical_and(in_x_range, in_y_range)
        n_gals_in_range_uvj = len(group_df[in_both_range_uvj])
        if paper_overview == False:
            ax.text(0.85, 0.85, f'{n_gals_in_range_uvj}', transform=ax.transAxes, fontsize=fontsize)
        # ax.plot(group_df['log_mass'], group_df['log_use_sfr'], marker='o', color='black', ls='None')

        if plot_row_idx in add_xaxis_labels:
            ax.set_xlabel('V-J', fontsize=fontsize)
        if plot_row_idx in add_yaxis_labels:
            ax.set_ylabel('U-V', fontsize=fontsize)
        if groupID != groupIDs[-1]:
            ax.tick_params(labelbottom=False)   
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.tick_params(labelsize = fontsize)
        scale_aspect(ax)




        ### BPT Plot --------------------------------------------
        ax = fig.add_subplot(gs[plot_row_idx, 4])
        xrange = (-2, 1)
        yrange = (-1.2, 1.5)
        
        plot_bpt(axis_obj=ax, skip_gals=True, add_background=True, snr_background=3)
        from bpt_clusters_singledf import add_composite_bpts
        add_composite_bpts(ax, groupID_toplot=groupID)

        try:
            group_df_bpt = plot_bpt(axis_obj=ax, use_other_df=1, use_df=group_df, add_background=False, color_gals=color_gals, add_prospector=prospector_run, groupID=groupID, plot_median=True, snr_background=3)
        except:
            group_df_bpt = plot_bpt(axis_obj=ax, use_other_df=1, use_df=group_df, add_background=False, color_gals=color_gals, plot_median=True, snr_background=3)
        in_x_range = np.logical_and(group_df_bpt['log_NII_Ha']>xrange[0], group_df_bpt['log_NII_Ha']<xrange[1])
        in_y_range = np.logical_and(group_df_bpt['log_OIII_Hb']>yrange[0], group_df_bpt['log_OIII_Hb']<yrange[1])
        in_both_range_bpt = np.logical_and(in_x_range, in_y_range)

        snr_background = 3
        group_df['hb_snr'] = group_df['hb_flux']/group_df['err_hb_flux']
        group_df['nii_6585_snr'] = group_df['nii_6585_flux']/group_df['err_nii_6585_flux']
        # group_df['ha_snr'] = group_df['ha_flux']/group_df['err_ha_flux']
        hb_detected = group_df['hb_snr']>snr_background
        nii_detected = group_df['nii_6585_snr']>snr_background
        both_detected = np.logical_and(hb_detected, nii_detected)
        n_galsbpt = len(group_df[both_detected])

        # n_gals_in_range_bpt = len(group_df[in_both_range_bpt])
        ax.text(0.85, 0.90, f'{n_galsbpt}', transform=ax.transAxes, fontsize=fontsize)
        # if paper_overview == False:
        #     ax.text(0.85, 0.85, f'{n_gals_in_range_bpt}', transform=ax.transAxes, fontsize=fontsize)
        # ax.plot(group_df['log_mass'], group_df['log_use_sfr'], marker='o', color='black', ls='None')

        # Add the measured value of the cluster
        # log_N2_Ha_group = clusters_summary_df['log_N2_Ha'].iloc[groupID]
        # log_O3_Hb_group = clusters_summary_df['log_O3_Hb'].iloc[groupID]
        # log_N2_Ha_group_errs = [[clusters_summary_df['err_log_N2_Ha_low'].iloc[groupID]], [clusters_summary_df['err_log_N2_Ha_high'].iloc[groupID]]]
        # log_O3_Hb_group_errs = [[clusters_summary_df['err_log_O3_Hb_low'].iloc[groupID]], [clusters_summary_df['err_log_O3_Hb_high'].iloc[groupID]]]

        # # ax.plot(log_N2_Ha_group, log_O3_Hb_group, marker=cluster_marker, color=get_row_color(groupID), markersize=get_row_size(i), mew=paper_marker_edge_width, mec=paper_mec, ls='None', zorder=10000, label='Composite')
        # rgba = get_row_color(groupID)
        # size = get_row_size(groupID)
        # # size = get_row_size(i)

        # if clusters_summary_df.iloc[groupID]['flag_hb_limit'] == 1:  
        #     log_O3_Hb_group = clusters_summary_df['log_O3_Hb_lower_limit'].iloc[groupID]
        #     rotation = 180
        #     # marker = (3, 0, rotation)
        #     arrow = u'$\u2191$'
        #     rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        #     rotated_marker._transform = rotated_marker.get_transform().rotate_deg(rotation)
        #     # markerfacecolor='None'
        #     ax.set_xlim(-2, 1)
        #     ax.set_ylim(-1.2, 1.5)

        #     xrange = 1 - (-2)
        #     yrange = 1.5 - (-1.2)
        #     arrow_width = 0.005
        #     arrow_length = 0.04

        #     ax.arrow((log_N2_Ha_group-(-2))/xrange, (log_O3_Hb_group-(-1.2))/yrange, 0, arrow_length, color=rgba, width=arrow_width, transform=ax.transAxes, zorder=1000000)
        #     ax.plot(log_N2_Ha_group, log_O3_Hb_group, marker='s', color=rgba, markersize=size, mec=paper_mec, mew=paper_marker_edge_width, ls='None', zorder=10000)
        # else:
        #     ax.errorbar(log_N2_Ha_group, log_O3_Hb_group, xerr=log_N2_Ha_group_errs, yerr=log_O3_Hb_group_errs, marker='s', color=rgba, markersize=size, mec=paper_mec, mew=paper_marker_edge_width, ls='None', zorder=10000)

        # ax.hlines(log_O3_Hb_group, log_N2_Ha_group-log_N2_Ha_group_errs[0], log_N2_Ha_group+log_N2_Ha_group_errs[1], color=get_row_color(groupID))
        # ax.vlines(log_N2_Ha_group, log_O3_Hb_group-log_O3_Hb_group_errs[0], log_O3_Hb_group+log_O3_Hb_group_errs[1], color=get_row_color(groupID))
        # ax.errorbar(log_N2_Ha_group, log_O3_Hb_group, xerr=log_N2_Ha_group_errs, yerr=log_O3_Hb_group_errs, marker='x', color='blue', markersize=10, mew=3, ls='None')
        # ax.errorbar(log_N2_Ha_group, log_O3_Hb_group, xerr=log_N2_Ha_group_errs, yerr=log_O3_Hb_group_errs, marker='o', color='blue')
        # Add the point from prospector

        if plot_row_idx in add_xaxis_labels:
            ax.set_xlabel('log(N[II]/H$\\alpha$)', fontsize=fontsize)
        if plot_row_idx in add_yaxis_labels:
            ax.set_ylabel('log(O[III]/H$\\beta$)', fontsize=fontsize, labelpad=-7)
        if groupID != groupIDs[-1]:
            ax.tick_params(labelbottom=False)   
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.tick_params(labelsize = fontsize)
        # ax.legend(framealpha=1)
        scale_aspect(ax)

        if paper_overview == False:
            ax.text(1.1, 0.3, f'Group {groupID}', transform=ax.transAxes, fontsize=20, rotation=270)

        plot_row_idx = plot_row_idx + 1
    # plt.tight_layout()
    if color_gals:
        save_str = '_color'
    else:
        save_str = ''
    if paper_overview==True:
        fig.savefig(imd.cluster_dir + f'/paper_figures/overview_clusters{save_str}{import_save_str}.pdf', bbox_inches='tight')
    else:
        fig.savefig(imd.cluster_dir + f'/cluster_stats/overview_clusters{save_str}{import_save_str}.pdf', bbox_inches='tight')


# setup_figs(20, norm_method='luminosity', paper_overview=False)
# make_overview_plot_clusters(20, bpt_color=True, paper_overview=True, prospector_spec=False)