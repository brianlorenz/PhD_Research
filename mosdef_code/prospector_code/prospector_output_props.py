#Manually import masses to save
import pandas as pd
import initialize_mosdef_dirs as imd
from astropy.io import ascii
import os
from balmer_avs import compute_balmer_av
from compute_cluster_sfrs import compute_cluster_sfrs
import sys
import numpy as np
from dust_equations_prospector import dust2_to_AV

def save_props(n_clusters, run_name):
    prop_dfs = []
    for groupID in range(n_clusters):
        prop_df = ascii.read(imd.prospector_fit_csvs_dir+f'/{run_name}_csvs/group{groupID}_props.csv').to_pandas()
        prop_df['groupID'] = groupID
        prop_dfs.append(prop_df)
    total_prop_df = pd.concat(prop_dfs)
    total_prop_df.to_csv(imd.prospector_output_dir + f'/{run_name}_props.csv', index=False)
    

def add_props_to_cluster_summary_df(n_clusters, run_name):
    total_prop_df = ascii.read(imd.prospector_output_dir + f'/{run_name}_props.csv').to_pandas()
    cluster_summary_df = imd.read_cluster_summary_df()
    # If there is no such column, merge the dataframes. If there is, check if we can do nothing or if instead we must update the column values
    if 'surviving_mass50' in cluster_summary_df.columns:
        print('Replacing old columns in cluster_summary_df')
        drop_colnames = total_prop_df.columns[total_prop_df.columns != 'groupID']
        cluster_summary_df = cluster_summary_df.drop(columns=drop_colnames)

    cluster_summary_df = cluster_summary_df.merge(total_prop_df, left_on='groupID', right_on='groupID')

    cluster_summary_df['Prospector_AV_16'] = dust2_to_AV(cluster_summary_df['dust2_16'])
    cluster_summary_df['Prospector_AV_50'] = dust2_to_AV(cluster_summary_df['dust2_50'])
    cluster_summary_df['Prospector_AV_84'] = dust2_to_AV(cluster_summary_df['dust2_84'])
    
    cluster_summary_df['AV_difference_with_limit'] = cluster_summary_df['balmer_av_with_limit'] - cluster_summary_df['Prospector_AV_50']
    cluster_summary_df['err_AV_difference_with_limit_low'] = np.sqrt(cluster_summary_df['err_balmer_av_with_limit_low']**2 + (cluster_summary_df['Prospector_AV_50']-cluster_summary_df['Prospector_AV_16'])**2)
    cluster_summary_df['err_AV_difference_with_limit_high'] = np.sqrt(cluster_summary_df['err_balmer_av_with_limit_high']**2 + (cluster_summary_df['Prospector_AV_84']-cluster_summary_df['Prospector_AV_50'])**2)

    cluster_summary_df['Prospector_ssfr50_target_mass'] = cluster_summary_df['sfr50'] / (10**cluster_summary_df['target_galaxy_median_log_mass'])
    cluster_summary_df['Prospector_ssfr50_normmedian_mass'] = cluster_summary_df['sfr50'] / (10**cluster_summary_df['norm_median_log_mass'])
    sfr50_norm = np.log10((cluster_summary_df['sfr50'] / (10**cluster_summary_df['norm_median_log_mass'])) * (10**cluster_summary_df['lum_weighted_median_log_mass']))
    cluster_summary_df['log_Prospector_ssfr50_multiplied_normalized'] = sfr50_norm
    cluster_summary_df['err_log_Prospector_ssfr50_multiplied_normalized_low'] = sfr50_norm - (np.log10((cluster_summary_df['sfr16'] / (10**cluster_summary_df['norm_median_log_mass'])) * (10**cluster_summary_df['lum_weighted_median_log_mass'])))
    cluster_summary_df['err_log_Prospector_ssfr50_multiplied_normalized_high'] = (np.log10((cluster_summary_df['sfr84'] / (10**cluster_summary_df['norm_median_log_mass'])) * (10**cluster_summary_df['lum_weighted_median_log_mass']))) - sfr50_norm
    sfr50_norm_target = np.log10((cluster_summary_df['sfr50'] / (10**cluster_summary_df['target_galaxy_median_log_mass'])) * (10**cluster_summary_df['lum_weighted_median_log_mass']))
    cluster_summary_df['log_Prospector_ssfr50_multiplied_normalized_targetmass'] = sfr50_norm_target
    cluster_summary_df['err_log_Prospector_ssfr50_multiplied_normalized_targetmass_low'] = sfr50_norm_target - (np.log10((cluster_summary_df['sfr16'] / (10**cluster_summary_df['target_galaxy_median_log_mass'])) * (10**cluster_summary_df['lum_weighted_median_log_mass'])))
    cluster_summary_df['err_log_Prospector_ssfr50_multiplied_normalized_targetmass_high'] = (np.log10((cluster_summary_df['sfr84'] / (10**cluster_summary_df['target_galaxy_median_log_mass'])) * (10**cluster_summary_df['lum_weighted_median_log_mass']))) - sfr50_norm_target

    cluster_summary_df['log_Prospector_ssfr50'] = np.log10(cluster_summary_df['sfr50'] / 10**cluster_summary_df['norm_median_log_mass'])
    cluster_summary_df['err_log_Prospector_ssfr50_low'] = cluster_summary_df['log_Prospector_ssfr50'] - np.log10(cluster_summary_df['sfr16'] / 10**cluster_summary_df['norm_median_log_mass'])
    cluster_summary_df['err_log_Prospector_ssfr50_high'] = np.log10(cluster_summary_df['sfr84'] / 10**cluster_summary_df['norm_median_log_mass']) - cluster_summary_df['log_Prospector_ssfr50']

    halphas = []
    err_halphas = []
    halpha_lums = []
    err_halpha_lums = []
    hbetas = []
    err_hbetas = []
    balmer_decs = []
    err_balmer_dec_lows = []
    err_balmer_dec_highs = []
    balmer_avs = []
    O3N2_metallicitys = []
    err_O3N2_metallicity_lows = []
    err_O3N2_metallicity_highs = []
    for groupID in range(n_clusters):
        emission_df_loc = imd.prospector_emission_fits_dir + f'/{run_name}_emission_fits/{groupID}_emission_fits.csv'
        pro_emission_df = ascii.read(emission_df_loc).to_pandas()
        ha_row = pro_emission_df[pro_emission_df['line_name']=='Halpha']
        hb_row = pro_emission_df[pro_emission_df['line_name']=='Hbeta']
        halpha = ha_row.iloc[0]['flux']
        err_halpha = ha_row.iloc[0]['err_flux']
        halpha_lum = ha_row.iloc[0]['luminosity']
        err_halpha_lum = ha_row.iloc[0]['err_luminosity']
        hbeta = hb_row.iloc[0]['flux']
        err_hbeta = hb_row.iloc[0]['err_flux']
        balmer_dec = pro_emission_df.iloc[0]['balmer_dec']
        err_balmer_dec_low = pro_emission_df.iloc[0]['err_balmer_dec_low']
        err_balmer_dec_high = pro_emission_df.iloc[0]['err_balmer_dec_high']
        balmer_av = compute_balmer_av(balmer_dec)
        O3N2_metallicity = pro_emission_df.iloc[0]['O3N2_metallicity']
        err_O3N2_metallicity_low = pro_emission_df.iloc[0]['err_O3N2_metallicity_low']
        err_O3N2_metallicity_high = pro_emission_df.iloc[0]['err_O3N2_metallicity_high']


        halphas.append(halpha)
        err_halphas.append(err_halpha)
        halpha_lums.append(halpha_lum)
        err_halpha_lums.append(err_halpha_lum)
        hbetas.append(hbeta)
        err_hbetas.append(err_hbeta)
        balmer_decs.append(balmer_dec)
        err_balmer_dec_lows.append(err_balmer_dec_low)
        err_balmer_dec_highs.append(err_balmer_dec_high)
        balmer_avs.append(balmer_av)
        O3N2_metallicitys.append(O3N2_metallicity)
        err_O3N2_metallicity_lows.append(err_O3N2_metallicity_low)
        err_O3N2_metallicity_highs.append(err_O3N2_metallicity_high)

    cluster_summary_df['prospector_log_mass'] = np.log10(cluster_summary_df['surviving_mass50'])
    cluster_summary_df['prospector_halpha_flux'] = halphas
    cluster_summary_df['err_prospector_halpha_flux'] = err_halphas
    cluster_summary_df['prospector_halpha_luminosity'] = halpha_lums
    cluster_summary_df['err_prospector_halpha_luminosity'] = err_halpha_lums
    cluster_summary_df['prospector_hbeta_flux'] = hbetas
    cluster_summary_df['err_prospector_hbeta_flux'] = err_hbetas
    cluster_summary_df['prospector_balmer_dec'] = balmer_decs
    cluster_summary_df['err_prospector_balmer_dec_low'] = err_balmer_dec_lows
    cluster_summary_df['err_prospector_balmer_dec_high'] = err_balmer_dec_highs
    cluster_summary_df['prospector_balmer_av'] = balmer_avs
    cluster_summary_df['prospector_O3N2_metallicity'] = O3N2_metallicitys
    cluster_summary_df['err_prospector_O3N2_metallicity_low'] = err_O3N2_metallicity_lows
    cluster_summary_df['err_prospector_O3N2_metallicity_high'] = err_O3N2_metallicity_highs

    cluster_summary_df['log_prospector_ssfr_prosmass_50'] = np.log10(cluster_summary_df['sfr50'] / cluster_summary_df['surviving_mass50'])
    cluster_summary_df['log_prospector_ssfr_prosmass_16'] = np.log10(cluster_summary_df['sfr16'] / cluster_summary_df['surviving_mass50'])
    cluster_summary_df['log_prospector_ssfr_prosmass_84'] = np.log10(cluster_summary_df['sfr84'] / cluster_summary_df['surviving_mass50'])
    cluster_summary_df['log_prospector_sfr_prosmass_50'] = np.log10(10**cluster_summary_df['log_prospector_ssfr_prosmass_50'] * 10**cluster_summary_df['lum_weighted_median_log_mass'])
    cluster_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)
    compute_cluster_sfrs(prospector=True)    

# save_props(20, 'metallicity_prior')
# add_props_to_cluster_summary_df(20, 'metallicity_prior')