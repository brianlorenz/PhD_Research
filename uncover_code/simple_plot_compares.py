import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import ascii
from plot_vals import scale_aspect, stellar_mass_label
from uncover_read_data import read_lineflux_cat, get_id_msa_list, read_SPS_cat, read_SPS_cat_all
from compute_av import compute_ha_pab_av, avneb_str, compute_ratio_from_av
import numpy as np

plot_shutter = True
phot_categories = False

def paper_plot_sed_emfit_accuracy(id_msa_list, color_var=''):
    full_data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.csv').to_pandas()
    data_df = full_data_df[full_data_df['id_msa'].isin(id_msa_list)]
    full_lineratio_data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df.csv').to_pandas()
    lineratio_data_df = full_lineratio_data_df[full_lineratio_data_df['id_msa'].isin(id_msa_list)]
    if plot_shutter:
        shutter_df = ascii.read('/Users/brianlorenz/uncover/Data/generated_tables/shutter_calcs.csv').to_pandas()
    sps_df = read_SPS_cat()
    sps_all_df = read_SPS_cat_all()

    # fig, axarr = plt.subplots(1,3,figsize=(18,6))
    fig = plt.figure(figsize=(18,6))
    fontsize = 16
    labelsize = 14
    ax_height = 0.66
    ax_width = 0.22
    ax_start_height = 0.15
    ax_ha_sed_vs_emfit = fig.add_axes([0.07, ax_start_height, ax_width, ax_height])
    ax_pab_sed_vs_emfit = fig.add_axes([0.36, ax_start_height, ax_width, ax_height])
    ax_av_sed_vs_emfit = fig.add_axes([0.65, ax_start_height, ax_width, ax_height])
    ax_list = [ax_ha_sed_vs_emfit, ax_pab_sed_vs_emfit]

    # line_p1 = np.array([-100, -100])
    # line_p2 = np.array([100, 100])
    line_p1 = np.array([-20, -20])
    line_p2 = np.array([-15, -15])
    def get_distance(datapoint):
        distance = np.cross(line_p2-line_p1,datapoint-line_p1)/np.linalg.norm(line_p2-line_p1)
        return distance

    markersize=8
    ecolor = 'gray'

    ha_distances = []
    pab_distances = []
    av_distances = []
    for id_msa in id_msa_list:
        if id_msa == 14880:
            continue
        data_df_row = data_df[data_df['id_msa'] == id_msa]
        lineratio_data_row = lineratio_data_df[lineratio_data_df['id_msa'] == id_msa]
        sps_row = sps_df[sps_df['id_msa']==id_msa]
        sps_all_row = sps_all_df[sps_all_df['id_msa']==id_msa]
        id_dr3 = sps_all_row['id'].iloc[0]
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        ha_snr = fit_df['signal_noise_ratio'].iloc[0]
        pab_snr = fit_df['signal_noise_ratio'].iloc[1]
        print(f'id_msa: {id_msa}, id_dr3 {id_dr3}')
        
        cmap = mpl.cm.inferno
        
        if color_var == 'sed_av':
            norm = mpl.colors.Normalize(vmin=0, vmax=3) 
            rgba = cmap(norm(lineratio_data_row['sed_av']))  
            cbar_label = 'Photometry AV'
        if color_var == 'ha_snr':
            norm = mpl.colors.LogNorm(vmin=2, vmax=100) 
            rgba = cmap(norm(ha_snr))
            cbar_label = 'H$\\alpha$ SNR'
        if color_var == 'pab_snr':
            norm = mpl.colors.Normalize(vmin=3, vmax=20) 
            rgba = cmap(norm(pab_snr))
            cbar_label = 'Pa$\\beta$ SNR'
        elif color_var == 'metallicity':
            norm = mpl.colors.Normalize(vmin=-1.2, vmax=0.1) 
            rgba = cmap(norm(sps_row['met_50']))
            cbar_label = 'Prospector Metallicity'
        elif color_var == 'mass':
            norm = mpl.colors.Normalize(vmin=7, vmax=10) 
            rgba = cmap(norm(sps_row['mstar_50']))
            cbar_label = stellar_mass_label
        if color_var != 'None':
            color_str = f'_{color_var}'
        else:
            color_str = ''

        # For photometry categories
        marker='o'
        av_marker='o'
        av_markersize = 8
        if phot_categories == True:
            ha_diff = False
            pab_diff = False
            if id_dr3 in [62937, 44283, 30804, 60579, 52140]:
                ha_diff = True
            if id_dr3 in [37776, 46339, 44283, 52140, 52257]:
                pab_diff = True
            if ha_diff and pab_diff:
                marker='X'
            elif ha_diff:
                marker='s'
            elif pab_diff:
                marker='v'
        

        


        # Emission fit data
        fit_ha_flux = fit_df['flux'].iloc[0]
        err_fit_ha_flux_low = fit_df['err_flux_low'].iloc[0]
        err_fit_ha_flux_high = fit_df['err_flux_high'].iloc[0]
        fit_pab_flux = fit_df['flux'].iloc[1]
        err_fit_pab_flux_low = fit_df['err_flux_low'].iloc[1]
        err_fit_pab_flux_high = fit_df['err_flux_high'].iloc[1]

        err_sed_ha_flux_low = lineratio_data_row['err_sed_ha_lineflux_low'].iloc[0]
        err_sed_ha_flux_high = lineratio_data_row['err_sed_ha_lineflux_high'].iloc[0]
        err_sed_fe_cor_pab_flux_low = lineratio_data_row['err_fe_cor_sed_pab_lineflux_low'].iloc[0]
        err_sed_fe_cor_pab_flux_high = lineratio_data_row['err_fe_cor_sed_pab_lineflux_high'].iloc[0]

        ha_datapoint = (fit_ha_flux, data_df_row['ha_sed_flux'].iloc[0])
        log_ha_datapoint = (np.log10(fit_ha_flux), np.log10(data_df_row['ha_sed_flux'].iloc[0]))
        ax_ha_sed_vs_emfit.errorbar(ha_datapoint[0], ha_datapoint[1], xerr=[[err_fit_ha_flux_low], [err_fit_ha_flux_high]], yerr=[[err_sed_ha_flux_low],[err_sed_ha_flux_high]], marker=marker, color=rgba, ls='None', mec='black', ms=markersize, ecolor=ecolor)
        ax_ha_sed_vs_emfit.set_xlabel('H$\\alpha$+NII Spectrum', fontsize=fontsize)
        ax_ha_sed_vs_emfit.set_ylabel('H$\\alpha$+NII Photometry', fontsize=fontsize)
        ha_distances.append(get_distance(np.array(log_ha_datapoint)))

        pab_datapoint = (fit_pab_flux, data_df_row['fe_cor_pab_sed_flux'].iloc[0])
        log_pab_datapoint = (np.log10(fit_pab_flux), np.log10(data_df_row['fe_cor_pab_sed_flux'].iloc[0]))
        ax_pab_sed_vs_emfit.errorbar(pab_datapoint[0], pab_datapoint[1], xerr=[[err_fit_pab_flux_low], [err_fit_pab_flux_high]], yerr=[[err_sed_fe_cor_pab_flux_low],[err_sed_fe_cor_pab_flux_high]], marker=marker, color=rgba, ls='None', mec='black', ms=markersize, ecolor=ecolor)
        ax_pab_sed_vs_emfit.set_xlabel('Pa$\\beta$ Spectrum', fontsize=fontsize)
        ax_pab_sed_vs_emfit.set_ylabel('Pa$\\beta$ Photometry', fontsize=fontsize)
        pab_distances.append(get_distance(np.array(log_pab_datapoint)))

        av_datapoint = np.array([lineratio_data_row['emission_fit_lineratio'].iloc[0], lineratio_data_row['sed_lineratio'].iloc[0]])
        inverse_av_datapoint = 1/av_datapoint
        av_datapoint_errs_low =np.array([lineratio_data_row['err_emission_fit_lineratio_low'].iloc[0], lineratio_data_row['err_sed_lineratio_low'].iloc[0]])
        av_datapoint_errs_high =np.array([lineratio_data_row['err_emission_fit_lineratio_high'].iloc[0], lineratio_data_row['err_sed_lineratio_high'].iloc[0]])
        inverse_av_datapoint_lows = 1/(av_datapoint-av_datapoint_errs_low)
        inverse_av_datapoint_highs = 1/(av_datapoint+av_datapoint_errs_high)
        err_inverse_av_datapoint_highs = np.abs(inverse_av_datapoint_lows - inverse_av_datapoint)
        err_inverse_av_datapoint_lows = np.abs(inverse_av_datapoint - inverse_av_datapoint_highs)
        # err_inverse_av_datapoint_highs = av_datapoint_errs_low / av_datapoint**2
        # err_inverse_av_datapoint_lows = av_datapoint_errs_high / av_datapoint**2

        # if id_msa ==  38163:
        #     breakpoint()
        
        # if av_datapoint_errs_high[1] > 50:
        #     av_marker=(3, 0, 180)
        #     err_inverse_av_datapoint_lows[1] = 0
        #     av_markersize=12
        # else:
        #     av_marker='o'
        #     av_markersize=8

        if plot_shutter == False:
            ax_av_sed_vs_emfit.errorbar(inverse_av_datapoint[0], inverse_av_datapoint[1], xerr=[[err_inverse_av_datapoint_lows[0]], [err_inverse_av_datapoint_highs[0]]], yerr=[[err_inverse_av_datapoint_lows[1]], [err_inverse_av_datapoint_highs[1]]], marker=marker, color=rgba, ls='None', mec='black', ms=av_markersize, ecolor=ecolor)
            ax_av_sed_vs_emfit.set_xlabel(f'(Pa$\\beta$ / H$\\alpha$) Spectrum', fontsize=fontsize)
            ax_av_sed_vs_emfit.set_ylabel(f'(Pa$\\beta$ / H$\\alpha$) Photometry', fontsize=fontsize)
            add_str=''
            av_value_datapoint = np.array([lineratio_data_row['emission_fit_av'].iloc[0], lineratio_data_row['sed_av'].iloc[0]])
            av_distances.append(get_distance(np.array(av_value_datapoint)))

        else:
            shutter_df_row = shutter_df[shutter_df['id_msa'] == id_msa]
            ax_av_sed_vs_emfit.errorbar(inverse_av_datapoint[0], shutter_df_row['lineratio_shutter'].iloc[0], xerr=[[err_inverse_av_datapoint_lows[0]], [err_inverse_av_datapoint_highs[0]]], marker=av_marker, color=rgba, ls='None', mec='black', ms=av_markersize, ecolor=ecolor)
            ax_av_sed_vs_emfit.set_xlabel(f'(Pa$\\beta$ / H$\\alpha$) Spectrum', fontsize=fontsize)
            ax_av_sed_vs_emfit.set_ylabel(f'(Pa$\\beta$ / H$\\alpha$) Photometry Shutter', fontsize=fontsize)
            add_str = '_shutter'
            av_value_datapoint = np.array([lineratio_data_row['emission_fit_av'].iloc[0], shutter_df_row['av_shutter'].iloc[0]])
            if (shutter_df_row['lineratio_shutter'].iloc[0]>0):
                av_distances.append(get_distance(np.array(av_value_datapoint)))
        

        add_text = 0
        if add_text:
            sps_total = read_SPS_cat_all()
            id_dr3 = sps_total[sps_total['id_msa'] == id_msa]['id'].iloc[0]
            ax_av_sed_vs_emfit.text(inverse_av_datapoint[0], inverse_av_datapoint[1], f'{id_dr3}')
            ax_pab_sed_vs_emfit.text(pab_datapoint[0], pab_datapoint[1], f'{id_dr3}')
            ax_ha_sed_vs_emfit.text(ha_datapoint[0], ha_datapoint[1], f'{id_dr3}')

    ha_distances = np.abs(ha_distances)
    pab_distances = np.abs(pab_distances)
    av_distances = np.abs(av_distances)
    median_ha_offset = np.median(ha_distances)
    scatter_ha_offset = np.std(ha_distances)
    median_pab_offset = np.median(pab_distances)
    scatter_pab_offset = np.std(pab_distances)
    median_av_offset = np.median(av_distances)
    scatter_av_offset = np.std(av_distances)

    start_scatter_text_x = 0.02
    start_scatter_text_y = 0.94
    scatter_text_sep = 0.07
    ax_ha_sed_vs_emfit.text(start_scatter_text_x, start_scatter_text_y, f'Offset: {median_ha_offset:0.2f}', transform=ax_ha_sed_vs_emfit.transAxes, fontsize=12)
    ax_ha_sed_vs_emfit.text(start_scatter_text_x, start_scatter_text_y-scatter_text_sep, f'Scatter: {scatter_ha_offset:0.2f}', transform=ax_ha_sed_vs_emfit.transAxes, fontsize=12)
    ax_pab_sed_vs_emfit.text(start_scatter_text_x, start_scatter_text_y, f'Offset: {median_pab_offset:0.2f}', transform=ax_pab_sed_vs_emfit.transAxes, fontsize=12)
    ax_pab_sed_vs_emfit.text(start_scatter_text_x, start_scatter_text_y-scatter_text_sep, f'Scatter: {scatter_pab_offset:0.2f}', transform=ax_pab_sed_vs_emfit.transAxes, fontsize=12)
    ax_av_sed_vs_emfit.text(start_scatter_text_x, start_scatter_text_y, f'AV Offset: {median_av_offset:0.2f}', transform=ax_av_sed_vs_emfit.transAxes, fontsize=12)
    ax_av_sed_vs_emfit.text(start_scatter_text_x, start_scatter_text_y-scatter_text_sep, f'Scatter: {scatter_av_offset:0.2f}', transform=ax_av_sed_vs_emfit.transAxes, fontsize=12)
    # ax_av_sed_vs_emfit.text(inverse_av_datapoint[0], inverse_av_datapoint[1], f'{id_msa}')
    # ax_pab_sed_vs_emfit.text(pab_datapoint[0], pab_datapoint[1], f'{id_msa}')
    
    
    if phot_categories == True:
        from matplotlib.lines import Line2D
        line_s = Line2D([0, 1], [0, 1], color=cmap(0.7), marker='s', ls='None')
        line_v = Line2D([0, 1], [0, 1], color=cmap(0.7), marker='v', ls='None')
        line_X = Line2D([0, 1], [0, 1], color=cmap(0.7), marker='X', ls='None')
        custom_lines_ha = [line_s, line_v, line_X]
        custom_labels_ha = ['Halpha offset', 'Pabeta offset', 'Both offset']
        ax_ha_sed_vs_emfit.legend(custom_lines_ha, custom_labels_ha, loc=3, fontsize=12)


    for ax in ax_list:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=labelsize)
        ax.plot([1e-20, 1e-14], [1e-20, 1e-14], ls='--', color='red', marker='None')
        ax.set_xlim([5e-19, 1e-15])
        ax.set_ylim([5e-19, 1e-15])
        # scale_aspect(ax)

    # Duplicating y axis
    
    ax2 = ax_av_sed_vs_emfit.twinx()
    ax_av_sed_vs_emfit.tick_params(labelsize=labelsize)
    ax2.tick_params(labelsize=labelsize)
    ax_av_sed_vs_emfit.plot([-10, 100], [-10, 100], ls='--', color='red', marker='None')
    # ax_av_sed_vs_emfit.set_xlim([38, 1.5])
    # ax_av_sed_vs_emfit.set_ylim([38, 1.5])
    ax_av_sed_vs_emfit.set_xlim([1/40, 1/1.5])
    ax_av_sed_vs_emfit.set_ylim([1/40, 1/1.5])
    ax2.set_ylim([38, 1.5])
    ax_av_sed_vs_emfit.set_xscale('log')
    ax_av_sed_vs_emfit.set_yscale('log')
    ax2.set_yscale('log')
    y_tick_locs = [0.03, 0.055, 1/10, 1/5, 1/2]
    y_tick_labs = ['0.03', '0.055', '0.1', '0.2', '0.5']
    ax_av_sed_vs_emfit.set_yticks(y_tick_locs)
    ax_av_sed_vs_emfit.set_yticklabels(y_tick_labs)
    ax_av_sed_vs_emfit.set_xticks(y_tick_locs)
    ax_av_sed_vs_emfit.set_xticklabels(y_tick_labs)
    # ax_av_sed_vs_emfit.set_xlim([-1.5, 4.5])
    # ax_av_sed_vs_emfit.set_ylim([-1.5, 4.5])
    twin_y_tick_labs = ['-1', '0', '1', '2', '3', '4']
    twin_y_tick_locs = [1/compute_ratio_from_av(int(rat)) for rat in twin_y_tick_labs]
    ax2.set_yticks(twin_y_tick_locs)
    ax2.set_yticklabels(twin_y_tick_labs)
    ax2.set_ylabel(f'Inferred {avneb_str}', fontsize=fontsize, rotation=270, labelpad=20)
    ax2.minorticks_off()
    ax_av_sed_vs_emfit.minorticks_off()
    # breakpoint()
    
    


    # scale_aspect(ax_av_sed_vs_emfit)

    cb_ax = fig.add_axes([.93, ax_start_height, .02, ax_height])
    if color_var != 'None':
        sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar_ticks = [7, 8, 9, 10, 11]
        cbar_ticklabels = [str(tick) for tick in cbar_ticks]
        cbar = fig.colorbar(sm, orientation='vertical', cax=cb_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticklabels)  
        cbar.set_label(cbar_label, fontsize=fontsize, rotation=270, labelpad=18)
        cbar.ax.tick_params(labelsize=fontsize)

    save_loc = f'/Users/brianlorenz/uncover/Figures/paper_figures/sed_vs_emfit{color_str}{add_str}.pdf'
    fig.savefig(save_loc, bbox_inches='tight')


def plot_simpletests(id_msa_list):
    full_data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.csv').to_pandas()
    data_df = full_data_df[full_data_df['id_msa'].isin(id_msa_list)]
    full_av_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df.csv').to_pandas()
    av_df = full_av_df[full_av_df['id_msa'].isin(id_msa_list)]

    fig, axarr = plt.subplots(2,4,figsize=(22,10))
    ax_ha_sed_vs_cat = axarr[0,0]
    ax_ha_sed_vs_emfit = axarr[0,1]
    ax_ha_cat_vs_emfit = axarr[0,2]
    ax_pab_sed_vs_cat = axarr[1,0]
    ax_pab_sed_vs_emfit = axarr[1,1]
    ax_pab_cat_vs_emfit = axarr[1,2]
    ax_ratio_cat_vs_emfit = axarr[0,3]
    ax_list = [ax_ha_sed_vs_cat, ax_ha_sed_vs_emfit, ax_ha_cat_vs_emfit, ax_pab_sed_vs_cat, ax_pab_sed_vs_emfit, ax_pab_cat_vs_emfit]

    sed_label = 'SED method'
    cat_label = 'UNCOVER Catalog'
    emfit_label = 'emission fit'


    ax_ha_sed_vs_cat.plot(data_df['ha_cat_flux'], data_df['ha_sed_flux'], marker='o', color='black', ls='None')
    ax_ha_sed_vs_cat.set_xlabel(cat_label)
    ax_ha_sed_vs_cat.set_ylabel(sed_label)

    ax_ha_sed_vs_emfit.plot(data_df['ha_emfit_flux'], data_df['ha_sed_flux'], marker='o', color='black', ls='None')
    ax_ha_sed_vs_emfit.set_xlabel(emfit_label)
    ax_ha_sed_vs_emfit.set_ylabel(sed_label)

    ax_ha_cat_vs_emfit.plot(data_df['ha_cat_flux'], data_df['ha_emfit_flux'], marker='o', color='black', ls='None')
    ax_ha_cat_vs_emfit.set_xlabel(cat_label)
    ax_ha_cat_vs_emfit.set_ylabel(emfit_label)

    ax_pab_sed_vs_cat.plot(data_df['pab_cat_flux'], data_df['pab_sed_flux'], marker='o', color='black', ls='None')
    ax_pab_sed_vs_cat.set_xlabel(cat_label)
    ax_pab_sed_vs_cat.set_ylabel(sed_label)

    ax_pab_sed_vs_emfit.plot(data_df['pab_emfit_flux'], data_df['pab_sed_flux'], marker='o', color='black', ls='None')
    ax_pab_sed_vs_emfit.set_xlabel(emfit_label)
    ax_pab_sed_vs_emfit.set_ylabel(sed_label)

    ax_pab_cat_vs_emfit.plot(data_df['pab_cat_flux'], data_df['pab_emfit_flux'], marker='o', color='black', ls='None')
    ax_pab_cat_vs_emfit.set_xlabel(cat_label)
    ax_pab_cat_vs_emfit.set_ylabel(emfit_label)

    cat_ratio = data_df['ha_cat_flux'] / data_df['pab_cat_flux']
    cat_av = compute_ha_pab_av(1/cat_ratio)
    emfit_ratio = data_df['ha_emfit_flux'] / data_df['pab_emfit_flux']
    emfit_av = compute_ha_pab_av(1/emfit_ratio)

    ax_ratio_cat_vs_emfit.plot(emfit_av, cat_av, marker='o', color='black', ls='None')
    for i in range(len(data_df)):
        if cat_av.iloc[i] > -2.5:
            ax_ha_cat_vs_emfit.text(data_df['ha_cat_flux'].iloc[i], data_df['ha_emfit_flux'].iloc[i], data_df['id_msa'].iloc[i])
            ax_pab_cat_vs_emfit.text(data_df['pab_cat_flux'].iloc[i], data_df['pab_emfit_flux'].iloc[i], data_df['id_msa'].iloc[i])
            ax_ratio_cat_vs_emfit.text(emfit_av.iloc[i], cat_av.iloc[i], data_df['id_msa'].iloc[i])

    ax_ratio_cat_vs_emfit.set_ylabel('Catalog Lineflux AV')
    ax_ratio_cat_vs_emfit.set_xlabel('Emission Fit AV')
    ax_ratio_cat_vs_emfit.tick_params(labelsize=12)
    ax_ratio_cat_vs_emfit.plot([-100, 100], [-100, 100], ls='--', color='red', marker='None')
    ax_ratio_cat_vs_emfit.set_xlim(-2.5, 2.5)
    ax_ratio_cat_vs_emfit.set_ylim(-2.5, 2.5)
    # scale_aspect(ax_ratio_cat_vs_emfit)


    for ax in ax_list:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=12)
        ax.plot([1e-20, 1e-15], [1e-20, 1e-15], ls='--', color='red', marker='None')
        ax.set_xlim([1e-20, 1e-15])
        ax.set_ylim([1e-20, 1e-15])
        # scale_aspect(ax)
        
    plt.tight_layout()
    save_loc = '/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/simpletest_flux_compare.pdf'
    fig.savefig(save_loc)

def plot_snr_compare(id_msa_list):
    fig, axarr = plt.subplots(1,2,figsize=(12,8))
    ax_ha_snr = axarr[0]
    ax_pab_snr = axarr[1]
    ax_list = [ax_ha_snr, ax_pab_snr]

    lines_df = read_lineflux_cat()


    for id_msa in id_msa_list:
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        ha_flux_fit = fit_df.iloc[0]['flux']
        pab_flux_fit = fit_df.iloc[1]['flux']
        ha_sigma = fit_df.iloc[0]['sigma'] # full width of the line
        pab_sigma = fit_df.iloc[1]['sigma'] # full width of the line
        ha_snr = fit_df['signal_noise_ratio'].iloc[0]
        pab_snr = fit_df['signal_noise_ratio'].iloc[1]

        lines_df_row = lines_df[lines_df['id_msa'] == id_msa]
        lines_df_ha_snr = lines_df_row['f_Ha+NII'].iloc[0] / lines_df_row['e_Ha+NII'].iloc[0]
        lines_df_pab_snr = lines_df_row['f_PaB'].iloc[0] / lines_df_row['e_PaB'].iloc[0]

        print(ha_snr)
    
        ax_ha_snr.plot(lines_df_ha_snr, ha_snr, marker='o', color='black', ls='None')
        ax_pab_snr.plot(lines_df_pab_snr, pab_snr, marker='o', color='black', ls='None')
    
    ax_ha_snr.set_xlabel('Catalog Ha SNR')
    ax_ha_snr.set_ylabel('Emfit Ha SNR')

    ax_pab_snr.set_xlabel('Catalog PaB SNR')
    ax_pab_snr.set_ylabel('Emfit PaB SNR')


    for ax in ax_list:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=12)
        ax.plot([-2, 10000], [-2, 10000], ls='--', color='red', marker='None')
        ax.set_xlim([0.01, 500])
        ax.set_ylim([0.01, 500])
        scale_aspect(ax)
        
    plt.tight_layout()
    save_loc = '/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/snr_compare_cat_emfit.pdf'
    fig.savefig(save_loc)

def plot_offsets(all=False):
    data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.csv')
    if all:
        data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.csv')

    fig, ax = plt.subplots(1,1,figsize=(6,6))
   
    # sed_label = 'SED method'
    # cat_label = 'UNCOVER Catalog'
    # emfit_label = 'emission fit'

    ax.plot(data_df['ha_sed_div_emfit'], data_df['pab_sed_div_emfit'], marker='o', color='black', ls='None')
    ax.set_xlabel('Ha offset sed/emfit')
    ax.set_ylabel('PaB offset sed/emfit')

    


    # for ax in ax_list:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=12)
    ax.plot([-10, 100], [-10, 100], ls='--', color='red', marker='None')
    ax.set_xlim([0.5, 40])
    ax.set_ylim([0.5, 40])
    
        
    plt.tight_layout()
    save_loc = '/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.pdf'
    if all:
        save_loc = '/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.pdf'
    fig.savefig(save_loc)

if __name__ == "__main__":
    id_msa_list = get_id_msa_list(full_sample=False)
    paper_plot_sed_emfit_accuracy(id_msa_list, color_var='mass')
    # plot_simpletests(id_msa_list)
    # plot_offsets(all=True)

    # id_msa_list = get_id_msa_list(full_sample=True)

    # plot_snr_compare(id_msa_list)