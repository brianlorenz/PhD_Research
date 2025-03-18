from full_phot_read_data import read_merged_lineflux_cat
from full_phot_merge_lineflux import filter_bcg_flags
import matplotlib.pyplot as plt
from uncover_read_data import read_SPS_cat_all, read_bcg_surface_brightness
from compute_av import compute_ha_pab_av, compute_pab_paa_av
import pandas as pd
import numpy as np
import random
from plot_vals import *

# Set thresholds here
snr_dict = {
    'Halpha_snr_thresh' : 2, 
    'PaBeta_snr_thresh' : 2, 
    'PaAlpha_snr_thresh' : 2, 
}
redshift_sigma_thresh = 1 # sigma or higher
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
    breakpoint()

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
    fig, ax = plt.subplots(figsize=(6,6))

    dfs = [lineflux_ha_pab, lineflux_pab_paa]
    colors = ['blue', 'red']
    labels = ['PaB/Ha AV', 'PaA/PaB AV']
    for i in range(len(dfs)):
        color = colors[i]
        df = dfs[i]
        label = labels[i]
        ax.plot(df['mstar_50'], df['AV_pab_ha'], marker='o', color=color, ls='None', label=label)

    # for i in range(len(lineflux_ha_pab)):
    #     plot_lineflux_ha_pab_err=np.array([[lineflux_ha_pab['err_AV_pab_ha_low'].iloc[i], lineflux_ha_pab['err_AV_pab_ha_high'].iloc[i]]]).T
    #     ax.errorbar(lineflux_ha_pab['mstar_50'].iloc[i], lineflux_ha_pab['AV_pab_ha'].iloc[i], yerr=plot_lineflux_ha_pab_err, marker='o', color='blue', ecolor='gray', ls='None', label='PaB/Ha AV')
    
    # for i in range(len(lineflux_pab_paa)):
    #     plot_lineflux_pab_paa_err=np.array([[lineflux_pab_paa['err_AV_paa_pab_low'].iloc[i], lineflux_pab_paa['err_AV_paa_pab_high'].iloc[i]]]).T
    #     ax.errorbar(lineflux_pab_paa['mstar_50'].iloc[i], lineflux_pab_paa['AV_paa_pab'].iloc[i], yerr=plot_lineflux_pab_paa_err, marker='o', color='red', ecolor='gray', ls='None', label='PaA/PaB AV')
    ax.legend()
    
    ax.set_xlabel(stellar_mass_label, fontsize=14)
    ax.set_ylabel(balmer_av_label, fontsize=14)
    save_str = ''
    save_str = add_thresh_text(ax)
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_analysis/dust_plots/mass_vs_av{save_str}.pdf')
    # plt.show()
    plt.close('all')

    # Sample select figure to match
    fig, ax = plt.subplots(figsize=(6,6))
    for i in range(len(dfs)):
        df = dfs[i]
        ax.plot(df['mstar_50'], df['z_50'], marker='o', color=colors[i], ls='None')
    ax.set_xlabel(stellar_mass_label, fontsize=14)
    ax.set_ylabel('Redshift', fontsize=14)
    save_str = add_thresh_text(ax)
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_analysis/sample_selects/sample_select{save_str}.pdf')
    # plt.show()
    plt.close('all')


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
    breakpoint()


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
    
    text_xstart = 0.03
    text_vstart = 0.9
    text_vsep = 0.08
    ax.text(text_xstart, text_vstart, f'Ha SNR > {ha_thresh}', transform=ax.transAxes)
    ax.text(text_xstart, text_vstart - text_vsep, f'PaB SNR > {pab_thresh}', transform=ax.transAxes)
    ax.text(text_xstart, text_vstart - 2*text_vsep, f'PaA SNR > {paa_thresh}', transform=ax.transAxes)
    ax.text(text_xstart, text_vstart - 3*text_vsep, f'z sigma > {redshift_sigma_thresh}', transform=ax.transAxes)
    ax.text(text_xstart, text_vstart - 4*text_vsep, f'bcg < {bcg_thresh}', transform=ax.transAxes)
    save_str = f'_bcg{bcg_thresh}_ha{ha_thresh}_pab{pab_thresh}_paa{paa_thresh}_zsig{redshift_sigma_thresh}'
    return save_str

if __name__ == "__main__":
    plot_mass_vs_dust()
    pass