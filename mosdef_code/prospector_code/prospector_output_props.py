#Manually import masses to save
import pandas as pd
import initialize_mosdef_dirs as imd
from astropy.io import ascii
import os
from balmer_avs import compute_balmer_av
from compute_cluster_sfrs import compute_cluster_sfrs
import sys
import numpy as np

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
        if cluster_summary_df['surviving_mass50'].iloc[0] != total_prop_df['surviving_mass50'].iloc[0]:
            sys.exit('Need to code this - new columns should REPLACE the old ones when merging')
        else:
            cluster_summary_df = cluster_summary_df.merge(total_prop_df, left_on='groupID', right_on='groupID')
    else:
        cluster_summary_df = cluster_summary_df.merge(total_prop_df, left_on='groupID', right_on='groupID')


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
    cluster_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)
    compute_cluster_sfrs(prospector=True)    

# save_props(20, 'metallicity_prior')
# add_props_to_cluster_summary_df(20, 'metallicity_prior')