from full_phot_read_data import read_merged_lineflux_cat
from full_phot_merge_lineflux import filter_bcg_flags
import matplotlib.pyplot as plt
from uncover_read_data import read_SPS_cat_all, read_bcg_surface_brightness
from compute_av import compute_ha_pab_av, compute_pab_paa_av
import pandas as pd
import numpy as np
import random
from plot_vals import *
import matplotlib as mpl
from matplotlib.lines import Line2D


# Set thresholds here
snr_dict = {
    'Halpha_snr_thresh' : 10, 
    'PaBeta_snr_thresh' : 10, 
    'PaAlpha_snr_thresh' : 10, 
}
redshift_sigma_thresh = 2 # sigma or higher
bcg_thresh = 0.04 # this value or lower


def plot_mass_vs_dust():
    # Set thresholds above

    lineflux_df = read_merged_lineflux_cat()    

    sps_df = read_SPS_cat_all()
    bcg_df = read_bcg_surface_brightness() 

    # Merge with sps
    lineflux_df = pd.merge(lineflux_df, sps_df, left_on='id_dr3', right_on='id', how='left')
    lineflux_df = pd.merge(lineflux_df, bcg_df, left_on='id_dr3', right_on='id_dr3', how='left')

    # Halpha to PaB dust measurement
    lineflux_ha_pab = lineflux_df[lineflux_df['lines_measured']==7]

    # PaB to PaA dust measurement
    lineflux_pab_paa = lineflux_df[lineflux_df['lines_measured']==9]

    lineflux_ha_pab = make_cuts_lineflux_df(lineflux_ha_pab, ha_pab_ratio=True)
    lineflux_pab_paa = make_cuts_lineflux_df(lineflux_pab_paa, pab_paa_ratio=True)
    
    # Compute dust measurements
    lineflux_ha_pab['AV_pab_ha'] = compute_ha_pab_av(lineflux_ha_pab['PaBeta_flux'] / lineflux_ha_pab['Halpha_flux'])
    err_av_low, err_av_high = calc_errs_worst(lineflux_ha_pab['AV_pab_ha'], lineflux_ha_pab['PaBeta_flux'], lineflux_ha_pab['err_PaBeta_flux_low'], lineflux_ha_pab['err_PaBeta_flux_high'], lineflux_ha_pab['Halpha_flux'], lineflux_ha_pab['err_Halpha_flux_low'], lineflux_ha_pab['err_Halpha_flux_high'])
    lineflux_ha_pab['err_AV_pab_ha_low'] = err_av_low
    lineflux_ha_pab['err_AV_pab_ha_high'] = err_av_high
    # plot_lineflux_ha_pab_err=np.array([[lineflux_ha_pab['err_AV_pab_ha_low'], lineflux_ha_pab['err_AV_pab_ha_high']]]).T
    # boot_errs(lineflux_ha_pab['AV_pab_ha'], lineflux_ha_pab['PaBeta_flux'], lineflux_ha_pab['err_PaBeta_flux_low'], lineflux_ha_pab['err_PaBeta_flux_high'], lineflux_ha_pab['Halpha_flux'], lineflux_ha_pab['err_Halpha_flux_low'], lineflux_ha_pab['err_Halpha_flux_high'])
   
    lineflux_pab_paa['AV_paa_pab'] = compute_pab_paa_av(lineflux_pab_paa['PaAlpha_flux'] / lineflux_pab_paa['PaBeta_flux'])
    err_av_low, err_av_high = calc_errs_worst(lineflux_pab_paa['AV_paa_pab'], lineflux_pab_paa['PaAlpha_flux'], lineflux_pab_paa['err_PaAlpha_flux_low'], lineflux_pab_paa['err_PaAlpha_flux_high'], lineflux_pab_paa['PaBeta_flux'], lineflux_pab_paa['err_PaBeta_flux_low'], lineflux_pab_paa['err_PaBeta_flux_high'])
    lineflux_pab_paa['err_AV_paa_pab_low'] = err_av_low
    lineflux_pab_paa['err_AV_paa_pab_high'] = err_av_high
    # plot_lineflux_pab_paa_err=np.array([[lineflux_pab_paa['err_AV_paa_pab_low'], lineflux_pab_paa['err_AV_paa_pab_high']]]).T
    # Bootstrap errors
    
    # Dust figure
    fig, ax = plt.subplots(figsize=(7,6))

    dfs = [lineflux_ha_pab, lineflux_pab_paa]
    colors = ['blue', 'red']
    shapes = ['o', 's']
    labels = ['PaB/Ha AV', 'PaA/PaB AV']
    av_names = ['AV_pab_ha', 'AV_paa_pab']
    line_names = [('Halpha', 'PaBeta'), ('PaBeta', 'PaAlpha')]
    
    cmap = mpl.cm.inferno
    norm = mpl.colors.LogNorm(vmin=0.5, vmax=50) 

    for i in range(len(dfs)):
        color = colors[i]
        df = dfs[i]
        label = labels[i]
        # Compute snr for coloring
        snrs = np.min([df[f'{line_names[i][0]}_snr'], df[f'{line_names[i][1]}_snr']], axis=0)
        df['min_snr'] = snrs
        breakpoint()
        for j in range(len(df)):
            rgba = cmap(norm(df['min_snr'].iloc[j]))
            # ax.errorbar(df['mstar_50'].iloc[j], df['z_50'].iloc[j], yerr=np.array([[low_zerr.iloc[j], high_zerr.iloc[j]]]).T, marker=shapes[i], mec='black', ms=6, color=rgba, ls='None', ecolor='gray')
            ax.plot(df['mstar_50'].iloc[j], df[av_names[i]].iloc[j], marker=shapes[i], color=rgba, markersize=6, mec='black', ls='None')
    # for i in range(len(lineflux_ha_pab)):
    #     plot_lineflux_ha_pab_err=np.array([[lineflux_ha_pab['err_AV_pab_ha_low'].iloc[i], lineflux_ha_pab['err_AV_pab_ha_high'].iloc[i]]]).T
    #     ax.errorbar(lineflux_ha_pab['mstar_50'].iloc[i], lineflux_ha_pab['AV_pab_ha'].iloc[i], yerr=plot_lineflux_ha_pab_err, marker='o', color='blue', ecolor='gray', ls='None', label='PaB/Ha AV')
    
    # for i in range(len(lineflux_pab_paa)):
    #     plot_lineflux_pab_paa_err=np.array([[lineflux_pab_paa['err_AV_paa_pab_low'].iloc[i], lineflux_pab_paa['err_AV_paa_pab_high'].iloc[i]]]).T
    #     ax.errorbar(lineflux_pab_paa['mstar_50'].iloc[i], lineflux_pab_paa['AV_paa_pab'].iloc[i], yerr=plot_lineflux_pab_paa_err, marker='o', color='red', ecolor='gray', ls='None', label='PaA/PaB AV')
    
    line_circles = Line2D([0], [0], color='orange', marker='o', markersize=6, ls='None', mec='black')
    line_squares = Line2D([0], [0], color='orange', marker='s', markersize=6, ls='None', mec='black')
    custom_lines = [line_circles, line_squares]
    custom_labels = ['Pa$\\beta$/H$\\alpha$', 'Pa$\\alpha$/Pa$\\beta$']
    ax.legend(custom_lines, custom_labels, loc=3, bbox_to_anchor=(1.05, 1.14))
    add_snr_cbar(fig, ax, norm, cmap)
    ax.set_xlabel(stellar_mass_label, fontsize=14)
    ax.set_ylabel('Inferred' + balmer_av_label, fontsize=14)
    ax.set_ylim(-20, 15)
    ax.set_xlim(5.5, 11.5)
    save_str = ''
    save_str = add_thresh_text(ax)
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_analysis/dust_plots/mass_vs_av{save_str}.pdf')
    # plt.show()
    plt.close('all')

    # Sample select figure to match
    fig, ax = plt.subplots(figsize=(7,6))
    for i in range(len(dfs)):
        df = dfs[i]
        # ax.plot(df['mstar_50'], df['z_50'], marker='o', color=colors[i], ls='None')
        # Compute errs
        low_zerr = df['z_50'] - df['z_16']
        high_zerr = df['z_84'] - df['z_50']
        
        
        
        for j in range(len(df)):
            rgba = cmap(norm(df['min_snr'].iloc[j]))
            ax.errorbar(df['mstar_50'].iloc[j], df['z_50'].iloc[j], yerr=np.array([[low_zerr.iloc[j], high_zerr.iloc[j]]]).T, marker=shapes[i], mec='black', ms=6, color=rgba, ls='None', ecolor='gray')
    ax.set_xlabel(stellar_mass_label, fontsize=14)
    ax.set_ylabel('Redshift', fontsize=14)
    ax.tick_params(labelsize=14)
    save_str = add_thresh_text(ax)
    add_snr_cbar(fig, ax, norm, cmap)
    line_circles = Line2D([0], [0], color='orange', marker='o', markersize=6, ls='None', mec='black')
    line_squares = Line2D([0], [0], color='orange', marker='s', markersize=6, ls='None', mec='black')
    custom_lines = [line_circles, line_squares]
    custom_labels = ['Pa$\\beta$/H$\\alpha$', 'Pa$\\alpha$/Pa$\\beta$']
    ax.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 1.14))
    ax.set_ylim(0, 2.5)
    ax.set_xlim(5.5, 11.5)
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_analysis/sample_selects/sample_select{save_str}.pdf')
    scale_aspect(ax)
    plt.close('all')


def add_snr_cbar(fig, ax, norm, cmap):
    #SNR cbar
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ticks = [0.5, 2, 5, 10, 50]
    cbar_ticklabels = [str(tick) for tick in cbar_ticks]
    cbar = fig.colorbar(sm, ax=ax, ticks=cbar_ticks)
    cbar.set_label('min(SNR)', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_yticklabels(cbar_ticklabels) 

def calc_errs_worst(av, line_num_flux, line_num_err_low, line_num_err_high, line_den_flux, line_den_err_low, line_den_err_high):
    pabeta_value_low = line_num_flux - line_num_err_low
    pabeta_value_high = line_num_flux + line_num_err_high
    halpha_value_low = line_den_flux - line_den_err_low
    halpha_value_high = line_den_flux + line_den_err_high
    av_value_low = compute_ha_pab_av(pabeta_value_low / halpha_value_high)
    av_value_high = compute_ha_pab_av(pabeta_value_high / halpha_value_low)
    err_av_low = av - av_value_low
    err_av_high = av_value_high - av
    return err_av_low, err_av_high

def boot_errs(av, line_num_flux, line_num_err_low, line_num_err_high, line_den_flux, line_den_err_low, line_den_err_high):
    # Currently having trouble bootstrapping if any go negative, since you can't toake log of a negative number
    # Needs to handle nondetections
    boot_vals = []
    for i in range(10):
        x = random.uniform(0, 1)
        if x < 0.5:
            boot_num = line_num_flux - np.abs(np.random.normal(loc = 0, scale=line_num_err_low))
        if x > 0.5:
            boot_num = np.abs(np.random.normal(loc = 0, scale=line_num_err_high)) + line_num_flux
        x = random.uniform(0, 1)
        if x < 0.5:
            boot_den = line_den_flux - np.abs(np.random.normal(loc = 0, scale=line_den_err_low))
        if x > 0.5:
            boot_den = np.abs(np.random.normal(loc = 0, scale=line_den_err_high)) + line_den_flux
        
        boot_val = compute_ha_pab_av(boot_num / boot_den)
        boot_vals.append(boot_val)
        if i == 0:
            df = boot_val.to_frame()
        else:
            df[f'{i}'] = boot_val


def make_cuts_lineflux_df(df, ha_pab_ratio=False, pab_paa_ratio=False):
    included_lines = []
    if ha_pab_ratio:
        included_lines = ['Halpha', 'PaBeta']
    if pab_paa_ratio:
        included_lines = ['PaBeta', 'PaAlpha']

    for line_name in included_lines:  # Loop through included lines
        # SNR cut  
        snr_thresh = snr_dict[f'{line_name}_snr_thresh']
        print(f'Making cuts {line_name}, with snr_thresh of {snr_thresh} and redshift_sigma of {redshift_sigma_thresh}')
        df = df[df[f'{line_name}_snr']>snr_thresh]
        # Redshift sig cut
        df = df[df[f'{line_name}_redshift_sigma']>redshift_sigma_thresh]
    # bcg cut, need to be less than threshold. Doesn't vary by line
    print(f'Making bcg cuts with surface brightness above {bcg_thresh}')
    df = df[df[f'bcg_surface_brightness']<bcg_thresh]

    return df

def add_thresh_text(ax):
    ha_thresh = snr_dict['Halpha_snr_thresh']
    pab_thresh = snr_dict['PaBeta_snr_thresh']
    paa_thresh = snr_dict['PaAlpha_snr_thresh']
    
    text_xstart = 0.01
    text_vstart = 1.11
    text_vsep = 0.04
    text_hsep = 0.35
    ax.text(text_xstart, text_vstart, f'Ha SNR   > {ha_thresh}', transform=ax.transAxes, color='cornflowerblue')
    ax.text(text_xstart, text_vstart - text_vsep, f'PaB SNR > {pab_thresh}', transform=ax.transAxes, color='blue')
    ax.text(text_xstart, text_vstart - 2*text_vsep, f'PaA SNR > {paa_thresh}', transform=ax.transAxes, color='darkblue')
    ax.text(text_xstart+text_hsep, text_vstart, f'z sigma > {redshift_sigma_thresh}', transform=ax.transAxes, color='mediumseagreen')
    ax.text(text_xstart+text_hsep, text_vstart - text_vsep, f'bcg       < {bcg_thresh}', transform=ax.transAxes, color='darkgreen')
    save_str = f'_bcg{bcg_thresh}_ha{ha_thresh}_pab{pab_thresh}_paa{paa_thresh}_zsig{redshift_sigma_thresh}'
    return save_str

if __name__ == "__main__":
    plot_mass_vs_dust()
    pass