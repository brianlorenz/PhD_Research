# Makes a datafrmae with summarized properties from all the clusters
import initialize_mosdef_dirs as imd
import numpy as np
from astropy.io import ascii
import pandas as pd
from axis_ratio_helpers import bootstrap_median

def make_clusters_summary_df(n_clusters, ignore_groups):
    """Makes a datafrmae with summarized properties from all the clusters
    
    Parameters:
    n_clusters (int): Number of clusters
    """
    
    groupIDs = []
    n_galss = []

    median_zs = []

    median_uvs = []
    median_vjs = []

    median_masses = []
    median_sfrs = []
    median_ssfrs = []

    av_medians = []
    err_av_median_lows = []
    err_av_median_highs = []
    beta_medians = []
    err_beta_median_lows = []
    err_beta_median_highs = []

    balmer_decs = []
    err_balmer_dec_lows = []
    err_balmer_dec_highs = []

    balmer_avs = []
    err_balmer_av_lows = []
    err_balmer_av_highs = []



    O3N2_metallicities = []
    err_O3N2_metallicity_lows = []
    err_O3N2_metallicity_highs = []

    log_N2_Has = []
    err_log_N2_Has_low = []
    err_log_N2_Has_high = []

    log_O3_Hbs = []
    err_log_O3_Hbs_low = []
    err_log_O3_Hbs_high = []

    avs = []
    betas = []

    

    
    for groupID in range(n_clusters):

        # Read in the df
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv')
        group_df['log_use_sfr'] = np.log10(group_df['use_sfr'])
        n_gals = len(group_df)


        # Compute properties
        median_z = np.median(group_df['Z_MOSFIRE'])
    
        median_vj = np.median(group_df[group_df['V_J']>0]['V_J'])
        median_uv = np.median(group_df[group_df['U_V']>0]['U_V'])

        median_mass = np.median(group_df[group_df['log_mass']>0]['log_mass'])
        median_sfr = np.median(group_df[group_df['log_use_sfr']>0]['log_use_sfr'])
        median_ssfr = np.log10((10**median_sfr)/(10**median_mass))

        # Save properties
        groupIDs.append(groupID)
        n_galss.append(n_gals)

        median_zs.append(median_z)

        median_vjs.append(median_vj)
        median_uvs.append(median_uv)

        median_masses.append(median_mass)
        median_sfrs.append(median_sfr)
        median_ssfrs.append(median_ssfr)

        av_median, err_av_median, err_av_median_low, err_av_median_high = bootstrap_median(group_df['AV'])
        beta_median, err_beta_median, err_beta_median_low, err_beta_median_high = bootstrap_median(group_df['beta'])
        av_medians.append(av_median)
        err_av_median_lows.append(err_av_median_low)
        err_av_median_highs.append(err_av_median_high)
        beta_medians.append(beta_median)
        err_beta_median_lows.append(err_beta_median_low)
        err_beta_median_highs.append(err_beta_median_high)

        # Read in the emission fits:
        # Once all groups are working, remove this if clause
        if groupID in ignore_groups:
            print(f'Ignoring group {groupID}')
            balmer_decs.append(-99)
            err_balmer_dec_lows.append(-99)
            err_balmer_dec_highs.append(-99)

            balmer_avs.append(-99)
            err_balmer_av_lows.append(-99)
            err_balmer_av_highs.append(-99)

            O3N2_metallicities.append(-99)
            err_O3N2_metallicity_lows.append(-99)
            err_O3N2_metallicity_highs.append(-99)

            log_N2_Has.append(-99)
            err_log_N2_Has_low.append(-99)
            err_log_N2_Has_high.append(-99)

            log_O3_Hbs.append(-99)
            err_log_O3_Hbs_low.append(-99)
            err_log_O3_Hbs_high.append(-99)

        else:
            emission_df = ascii.read(imd.emission_fit_csvs_dir + f'/{groupID}_emission_fits.csv').to_pandas()
            balmer_decs.append(emission_df.iloc[0]['balmer_dec'])
            err_balmer_dec_lows.append(emission_df.iloc[0]['err_balmer_dec_low'])
            err_balmer_dec_highs.append(emission_df.iloc[0]['err_balmer_dec_high'])

            balmer_av = 4.05*1.97*np.log10(emission_df.iloc[0]['balmer_dec']/2.86)
            # Recalculate where the errors would be if the points were at the top/bottom of their ranges
            err_balmer_av_low = balmer_av - 4.05*1.97*np.log10((emission_df.iloc[0]['balmer_dec']-emission_df.iloc[0]['err_balmer_dec_low'])/2.86)
            err_balmer_av_high = 4.05*1.97*np.log10((emission_df.iloc[0]['balmer_dec']+emission_df.iloc[0]['err_balmer_dec_high'])/2.86) - balmer_av
            balmer_avs.append(balmer_av)
            err_balmer_av_lows.append(err_balmer_av_low)
            err_balmer_av_highs.append(err_balmer_av_high)

            O3N2_metallicities.append(emission_df.iloc[0]['O3N2_metallicity'])
            err_O3N2_metallicity_lows.append(emission_df.iloc[0]['err_O3N2_metallicity_low'])
            err_O3N2_metallicity_highs.append(emission_df.iloc[0]['err_O3N2_metallicity_high'])

            log_N2_Has.append(emission_df.iloc[0]['log_N2_Ha'])
            err_log_N2_Has_low.append(emission_df.iloc[0]['err_log_N2_Ha_low'])
            err_log_N2_Has_high.append(emission_df.iloc[0]['err_log_N2_Ha_high'])

            log_O3_Hbs.append(emission_df.iloc[0]['log_O3_Hb'])
            err_log_O3_Hbs_low.append(emission_df.iloc[0]['err_log_O3_Hb_low'])
            err_log_O3_Hbs_high.append(emission_df.iloc[0]['err_log_O3_Hb_high'])
            
            
    # Build into DataFrame
    clusters_summary_df = pd.DataFrame(zip(groupIDs, n_galss, median_zs, median_masses, median_sfrs, median_ssfrs, av_medians, err_av_median_lows, err_av_median_highs, beta_medians, err_beta_median_lows, err_beta_median_highs, median_vjs, median_uvs, balmer_decs, err_balmer_dec_lows, err_balmer_dec_highs, balmer_avs, err_balmer_av_lows, err_balmer_av_highs, O3N2_metallicities, err_O3N2_metallicity_lows, err_O3N2_metallicity_highs, log_N2_Has, err_log_N2_Has_low, err_log_N2_Has_high, log_O3_Hbs, err_log_O3_Hbs_low, err_log_O3_Hbs_high), columns=['groupID', 'n_gals', 'redshift', 'log_mass', 'log_sfr', 'log_ssfr', 'AV', 'err_AV_low', 'err_AV_high', 'beta', 'err_beta_low', 'err_beta_high', 'median_V_J', 'median_U_V', 'balmer_dec', 'err_balmer_dec_low', 'err_balmer_dec_high', 'balmer_av', 'err_balmer_av_low', 'err_balmer_av_high', 'O3N2_metallicity', 'err_O3N2_metallicity_low', 'err_O3N2_metallicity_high', 'log_N2_Ha', 'err_log_N2_Ha_low', 'err_log_N2_Ha_high', 'log_O3_Hb', 'err_log_O3_Hb_low', 'err_log_O3_Hb_high'])
    clusters_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)

# make_clusters_summary_df(23, ignore_groups=[19])