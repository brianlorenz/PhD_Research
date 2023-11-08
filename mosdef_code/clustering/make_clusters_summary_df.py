# Makes a datafrmae with summarized properties from all the clusters
import initialize_mosdef_dirs as imd
import numpy as np
from astropy.io import ascii
import pandas as pd
from axis_ratio_helpers import bootstrap_median
from cosmology_calcs import flux_to_luminosity

def make_clusters_summary_df(n_clusters, ignore_groups, use_ha_first_csvs=False, halpha_scaled=False):
    """Makes a datafrmae with summarized properties from all the clusters
    
    Parameters:
    n_clusters (int): Number of clusters
    use_ha_first_csvs (boolean): Set to true to use the emission fits that fit halpha first
    halpha_scaled (boolean): Set to true to use the halpha scaled emission fits
    """
    
    groupIDs = []
    n_galss = []
    halpha_scaled_values = []

    median_zs = []
    weighted_median_zs = []

    median_uvs = []
    median_vjs = []

    median_masses = []
    mean_masses = []
    norm_median_masses = []
    median_sfrs = []
    median_ssfrs = []
    median_res = []
    median_halphas = []
    norm_median_halphas = []
    median_halpha_lums = []

    av_medians = []
    err_av_median_lows = []
    err_av_median_highs = []
    beta_medians = []
    err_beta_median_lows = []
    err_beta_median_highs = []

    ha_fluxes = []
    err_ha_fluxes = []
    hb_fluxes = []
    err_hb_fluxes = []
    hb_sns = []

    balmer_decs = []
    err_balmer_dec_lows = []
    err_balmer_dec_highs = []
    balmer_dec_sns = []

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

    computed_ssfrs = []

    

    
    for groupID in range(n_clusters):

        # Read in the df
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        group_df['log_use_sfr'] = np.log10(group_df['use_sfr'])
        n_gals = len(group_df)
        halpha_scaled_value = halpha_scaled

        def weighted_quantiles(values, weights, quantiles=0.5):
            i = np.argsort(values)
            c = np.cumsum(weights[i])
            return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]

        # Compute properties
        median_z = np.median(group_df['Z_MOSFIRE'])
        group_df['zero_ha_flux'] = group_df['ha_flux']
        group_df.loc[group_df['zero_ha_flux'] == -999.0, 'zero_ha_flux'] = 0
        weighted_median_z = weighted_quantiles(group_df['Z_MOSFIRE'].to_numpy(), group_df['zero_ha_flux'].to_numpy())
        median_vj = np.median(group_df[group_df['V_J']>0]['V_J'])
        median_uv = np.median(group_df[group_df['U_V']>0]['U_V'])

        median_mass = np.median(group_df[group_df['log_mass']>0]['log_mass'])
        mean_mass = np.mean(group_df[group_df['log_mass']>0]['log_mass'])
        # Compute norm median mass
        group_masses = 10**(group_df[group_df['log_mass']>0]['log_mass'])
        norm_group_masses = group_df[group_df['log_mass']>0]['norm_factor'] * group_masses
        norm_median_mass = np.median(np.log10(norm_group_masses))
        median_sfr = np.median(group_df[group_df['log_use_sfr']>0]['log_use_sfr'])
        median_ssfr = np.log10((10**median_sfr)/(10**median_mass))
        re_median = np.median(group_df['half_light'])
        median_halpha = np.median(group_df[group_df['ha_flux']>0]['ha_flux'])
        norm_median_halpha = np.median(group_df[group_df['ha_flux']>0]['ha_flux'] * group_df[group_df['ha_flux']>0]['norm_factor'])
        median_halpha_lum = np.median(flux_to_luminosity(group_df[group_df['ha_flux']>0]['ha_flux'], group_df[group_df['ha_flux']>0]['Z_MOSFIRE']))


        # Save properties
        groupIDs.append(groupID)
        n_galss.append(n_gals)
        halpha_scaled_values.append(halpha_scaled_value)

        median_zs.append(median_z)
        weighted_median_zs.append(weighted_median_z)

        median_vjs.append(median_vj)
        median_uvs.append(median_uv)

        median_masses.append(median_mass)
        mean_masses.append(mean_mass)
        norm_median_masses.append(norm_median_mass)
        median_sfrs.append(median_sfr)
        median_ssfrs.append(median_ssfr)
        median_res.append(re_median)
        median_halphas.append(median_halpha)
        norm_median_halphas.append(norm_median_halpha)
        median_halpha_lums.append(median_halpha_lum)

        av_median, err_av_median, err_av_median_low, err_av_median_high = bootstrap_median(group_df['norm_factor'] * group_df['AV'])
        beta_median, err_beta_median, err_beta_median_low, err_beta_median_high = bootstrap_median(group_df['norm_factor'] * group_df['beta'])
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
            ha_fluxes.append(-99)
            err_ha_fluxes.append(-99)
            hb_fluxes.append(-99)
            err_hb_fluxes.append(-99)
            hb_sns.append(-99)

            balmer_decs.append(-99)
            err_balmer_dec_lows.append(-99)
            err_balmer_dec_highs.append(-99)
            balmer_dec_sns.append(-99)

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

            computed_ssfrs.append(-99)



        else:
            if use_ha_first_csvs==True:
                emission_df = ascii.read(imd.emission_fit_dir + '/ha_first_csvs/' + f'/{groupID}_emission_fits.csv').to_pandas()
            else:
                if halpha_scaled == True:
                    emission_df = ascii.read(imd.emission_fit_dir + f'/halpha_scaled_emission_fitting_csvs/{groupID}_emission_fits.csv').to_pandas()
                else:
                    emission_df = ascii.read(imd.emission_fit_csvs_dir + f'/{groupID}_emission_fits.csv').to_pandas()
            
            ha_row = emission_df[emission_df['line_name'] == 'Halpha']
            ha_fluxes.append(ha_row.iloc[0]['flux'])
            err_ha_fluxes.append(ha_row.iloc[0]['err_flux'])
            hb_row = emission_df[emission_df['line_name'] == 'Hbeta']
            hb_fluxes.append(hb_row.iloc[0]['flux'])
            err_hb_fluxes.append(hb_row.iloc[0]['err_flux'])
            hb_sns.append(hb_row.iloc[0]['flux'] / hb_row.iloc[0]['err_flux'])
            
            balmer_decs.append(emission_df.iloc[0]['balmer_dec'])
            err_balmer_dec_lows.append(emission_df.iloc[0]['err_balmer_dec_low'])
            err_balmer_dec_highs.append(emission_df.iloc[0]['err_balmer_dec_high'])
            balmer_dec_sns.append(emission_df.iloc[0]['balmer_dec'] / np.mean([emission_df.iloc[0]['err_balmer_dec_low'], emission_df.iloc[0]['err_balmer_dec_high']]))

            balmer_av = 4.05*1.97*np.log10(emission_df.iloc[0]['balmer_dec']/2.86)
            # Recalculate where the errors would be if the points were at the top/bottom of their ranges
            err_balmer_av_low = balmer_av - 4.05*1.97*np.log10((emission_df.iloc[0]['balmer_dec']-emission_df.iloc[0]['err_balmer_dec_low'])/2.86)
            err_balmer_av_high = 4.05*1.97*np.log10((emission_df.iloc[0]['balmer_dec']+emission_df.iloc[0]['err_balmer_dec_high'])/2.86) - balmer_av
            balmer_avs.append(balmer_av)
            err_balmer_av_lows.append(err_balmer_av_low)
            err_balmer_av_highs.append(err_balmer_av_high)

            try:
                O3N2_metallicities.append(emission_df.iloc[0]['O3N2_metallicity'])
                err_O3N2_metallicity_lows.append(emission_df.iloc[0]['err_O3N2_metallicity_low'])
                err_O3N2_metallicity_highs.append(emission_df.iloc[0]['err_O3N2_metallicity_high'])

                log_N2_Has.append(emission_df.iloc[0]['log_N2_Ha'])
                err_log_N2_Has_low.append(emission_df.iloc[0]['err_log_N2_Ha_low'])
                err_log_N2_Has_high.append(emission_df.iloc[0]['err_log_N2_Ha_high'])

                log_O3_Hbs.append(emission_df.iloc[0]['log_O3_Hb'])
                err_log_O3_Hbs_low.append(emission_df.iloc[0]['err_log_O3_Hb_low'])
                err_log_O3_Hbs_high.append(emission_df.iloc[0]['err_log_O3_Hb_high'])

            except:
                O3N2_metallicities.append(-99)
                err_O3N2_metallicity_lows.append(-99)
                err_O3N2_metallicity_highs.append(-99)

                log_N2_Has.append(-99)
                err_log_N2_Has_low.append(-99)
                err_log_N2_Has_high.append(-99)

                log_O3_Hbs.append(-99)
                err_log_O3_Hbs_low.append(-99)
                err_log_O3_Hbs_high.append(-99)

            
    # Build into DataFrame
    clusters_summary_df = pd.DataFrame(zip(groupIDs, n_galss, halpha_scaled_values, median_zs, weighted_median_zs, median_masses, mean_masses, norm_median_masses, median_sfrs, median_ssfrs, median_res, median_halphas, norm_median_halphas, median_halpha_lums, av_medians, err_av_median_lows, err_av_median_highs, beta_medians, err_beta_median_lows, err_beta_median_highs, median_vjs, median_uvs, ha_fluxes, err_ha_fluxes, hb_fluxes, err_hb_fluxes, hb_sns, balmer_decs, err_balmer_dec_lows, err_balmer_dec_highs, balmer_dec_sns, balmer_avs, err_balmer_av_lows, err_balmer_av_highs, O3N2_metallicities, err_O3N2_metallicity_lows, err_O3N2_metallicity_highs, log_N2_Has, err_log_N2_Has_low, err_log_N2_Has_high, log_O3_Hbs, err_log_O3_Hbs_low, err_log_O3_Hbs_high), columns=['groupID', 'n_gals', 'halpha_scaled_spectra', 'redshift', 'flux_weighted_redshift', 'median_log_mass', 'mean_log_mass', 'norm_median_log_mass', 'median_log_sfr', 'median_log_ssfr', 'median_re', 'median_indiv_halphas', 'norm_median_halphas', 'median_halpha_luminosity', 'AV', 'err_AV_low', 'err_AV_high', 'beta', 'err_beta_low', 'err_beta_high', 'median_V_J', 'median_U_V', 'ha_flux', 'err_ha_flux', 'hb_flux', 'err_hb_flux', 'hb_snr', 'balmer_dec', 'err_balmer_dec_low', 'err_balmer_dec_high', 'balmer_dec_snr', 'balmer_av', 'err_balmer_av_low', 'err_balmer_av_high', 'O3N2_metallicity', 'err_O3N2_metallicity_low', 'err_O3N2_metallicity_high', 'log_N2_Ha', 'err_log_N2_Ha_low', 'err_log_N2_Ha_high', 'log_O3_Hb', 'err_log_O3_Hb_low', 'err_log_O3_Hb_high'])
    clusters_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)

# make_clusters_summary_df(23, ignore_groups=[19], use_ha_first_csvs=False)