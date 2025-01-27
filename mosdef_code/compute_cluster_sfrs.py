import initialize_mosdef_dirs as imd
from cosmology_calcs import flux_to_luminosity
from compute_new_sfrs import correct_lum_for_dust, ha_lum_to_sfr, correct_av_for_dust, apply_dust_law
from balmer_avs import compute_balmer_av
import numpy as np
from astropy.io import ascii
import random
import pandas as pd

def compute_cluster_sfrs(lower_limit=True, luminosity=False, prospector=False, monte_carlo = True, bootstrap=-1):
    """
    
    Parameters:
    lower_limit (boolean): Set to true to use the lower limits computed in balmer_dec_histogram
    luminosity (boolean): Set to false if the fluxes are already in luminosity space
    prospector (boolean): Set to true if you want to compute the prospector SFRs
    monte_carlo (boolean): Set to true to use the monte-carlo emission fits to compute errors
    bootstrap (float): Number of boostrapped emission measurements - set to -1 to skip error calucation
    """
    cluster_summary_df = imd.read_cluster_summary_df()

    halpha_fluxes = cluster_summary_df['ha_flux']
    err_halpha_fluxes = cluster_summary_df['err_ha_flux']
    balmer_avs = cluster_summary_df['balmer_av']
    balmer_decs = cluster_summary_df['balmer_dec']
    err_balmer_avs_low = cluster_summary_df['err_balmer_av_low']
    err_balmer_avs_high = cluster_summary_df['err_balmer_av_high']
    if lower_limit == True:
        balmer_avs = cluster_summary_df['balmer_av_with_limit']
        balmer_decs = cluster_summary_df['balmer_dec_with_limit']
        err_balmer_avs_low = cluster_summary_df['err_balmer_av_with_limit_low']
        err_balmer_avs_high = cluster_summary_df['err_balmer_av_with_limit_high']

    # log_mean_masses = cluster_summary_df['mean_log_mass']
    

    # For redshifts, we now want to use the target galaxy rather than the average of the group
    # redshifts = cluster_summary_df['redshift']
    group_dfs = [ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas() for groupID in range(len(cluster_summary_df))]
    norm_gals = [group_dfs[i][group_dfs[i]['norm_factor']==1] for i in range(len(cluster_summary_df))]
    redshifts = [norm_gals[i]['Z_MOSFIRE'].iloc[0] for i in range(len(cluster_summary_df))]
    cluster_summary_df['target_galaxy_redshifts'] = redshifts
    cluster_summary_df['target_galaxy_median_log_mass'] = [norm_gals[i]['log_mass'].iloc[0] for i in range(len(cluster_summary_df))]
    
    median_redshifts = cluster_summary_df['redshift']
    weighted_median_redshifts = cluster_summary_df['flux_weighted_redshift']
    log_norm_median_masses = cluster_summary_df['norm_median_log_mass']
    log_median_masses = cluster_summary_df['median_log_mass']

    # Another method is to try to compute using A_V rather than mass
    AV = cluster_summary_df['AV']

    #Convert the Balmer AV to A_Halpha using https://iopscience.iop.org/article/10.1088/0004-637X/763/2/145/pdf Dominguez +2013
    balmer_ahalphas = compute_balmer_ahalpha_from_AV(balmer_avs, law='Cardelli')
    err_balmer_halphas_low = compute_balmer_ahalpha_from_AV(err_balmer_avs_low, law='Cardelli')
    err_balmer_halphas_high = compute_balmer_ahalpha_from_AV(err_balmer_avs_high, law='Cardelli')
    # ahalphas = 3.33*(AV / 4.05)*2

    # Convert ha to luminsoty
    halpha_lums, err_halpha_lums = flux_to_luminosity(halpha_fluxes, median_redshifts, err_halpha_fluxes)
    if luminosity == True:
        halpha_lums = halpha_fluxes
        err_halpha_lums = err_halpha_fluxes
    if prospector == True:
        prospector_halpha_lums = cluster_summary_df['prospector_halpha_luminosity']
        err_halpha_lums = cluster_summary_df['err_prospector_halpha_luminosity']
        prospector_log_median_masses = np.log10(cluster_summary_df['surviving_mass50'])
        prospector_balmer_avs = cluster_summary_df['prospector_balmer_av']
        prospector_balmer_ahalphas = compute_balmer_ahalpha_from_AV(prospector_balmer_avs, law='Cardelli')

    log_halpha_sfrs, log_halpha_ssfrs = perform_sfr_computation(halpha_lums, balmer_ahalphas, log_median_masses, imf='subsolar')
    if prospector == True:
        prospector_av_log_halpha_sfrs, prospector_av_log_halpha_ssfrs = perform_sfr_computation(prospector_halpha_lums, prospector_balmer_ahalphas, prospector_log_median_masses, imf='subsolar')


    # Monte carlo or bootstrap errors on sfr
    if monte_carlo == True:
        err_sfr_low, err_sfr_high, err_ssfr_low, err_ssfr_high = get_montecarlo_errs(log_median_masses, log_halpha_sfrs, log_halpha_ssfrs, median_redshifts, luminosity, imf='subsolar')
    elif bootstrap > 0:
        err_sfr_low, err_sfr_high, err_ssfr_low, err_ssfr_high = get_sfr_errs(bootstrap, halpha_lums, err_halpha_lums, balmer_ahalphas, err_balmer_halphas_low, err_balmer_halphas_high, log_median_masses, log_halpha_sfrs, log_halpha_ssfrs, median_redshifts, luminosity, imf='subsolar')
    else:
        err_sfr_low = -99
        err_sfr_high = -99
        err_ssfr_low = -99
        err_ssfr_high = -99


    if prospector == True:
        cluster_summary_df['prospector_log_sfr'] = prospector_av_log_halpha_sfrs
        cluster_summary_df['prospector_log_ssfr'] = prospector_av_log_halpha_ssfrs
        cluster_summary_df['cluster_av_prospector_sfr'] = log_halpha_sfrs
        cluster_summary_df['cluster_av_prospector_log_ssfr'] = log_halpha_ssfrs
    else:
        cluster_summary_df['computed_log_sfr'] = log_halpha_sfrs
        cluster_summary_df['err_computed_log_sfr_low'] = err_sfr_low
        cluster_summary_df['err_computed_log_sfr_high'] = err_sfr_high

        cluster_summary_df['computed_log_ssfr'] = log_halpha_ssfrs
        cluster_summary_df['err_computed_log_ssfr_low'] = err_ssfr_low
        cluster_summary_df['err_computed_log_ssfr_high'] = err_ssfr_high

        cluster_summary_df['computed_log_sfr_with_limit'] = -99
        cluster_summary_df['err_computed_log_sfr_with_limit_low'] = -99
        cluster_summary_df['err_computed_log_sfr_with_limit_high'] = -99

        cluster_summary_df['computed_log_ssfr_with_limit'] = -99
        cluster_summary_df['err_computed_log_ssfr_with_limit_low'] = -99
        cluster_summary_df['err_computed_log_ssfr_with_limit_high'] = -99

        if lower_limit==True:
            cluster_summary_df['computed_log_sfr'] = -99
            cluster_summary_df['err_computed_log_sfr_low'] = -99
            cluster_summary_df['err_computed_log_sfr_high'] = -99

            cluster_summary_df['computed_log_ssfr'] = -99
            cluster_summary_df['err_computed_log_ssfr_low'] = -99
            cluster_summary_df['err_computed_log_ssfr_high'] = -99

            cluster_summary_df['computed_log_sfr_with_limit'] = log_halpha_sfrs
            cluster_summary_df['err_computed_log_sfr_with_limit_low'] = err_sfr_low
            cluster_summary_df['err_computed_log_sfr_with_limit_high'] = err_sfr_high
            
            cluster_summary_df['computed_log_ssfr_with_limit'] = log_halpha_ssfrs
            cluster_summary_df['err_computed_log_ssfr_with_limit_low'] = err_ssfr_low
            cluster_summary_df['err_computed_log_ssfr_with_limit_high'] = err_ssfr_high

    

    
    cluster_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)

def compute_new_sfrs_compositepaper(n_clusters, imf='subsolar'):
    """Computes SFRs for all of the galaxies - using dust correction where possible, AV (with a factor) if not, and the same conversion as the composites"""
    for groupID in range(n_clusters):
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        halpha_fluxes = group_df['ha_flux']
        hbeta_fluxes = group_df['hb_flux']
        balmer_decs = halpha_fluxes / hbeta_fluxes
        redshifts = group_df['Z_MOSFIRE']
        AVs = group_df['AV']
        log_masses = group_df['log_mass']

        balmer_avs = compute_balmer_av(balmer_decs)
        balmer_ahalphas = compute_balmer_ahalpha_from_AV(balmer_avs, law='Cardelli')

        halpha_lums = flux_to_luminosity(halpha_fluxes, redshifts)

        log_balmer_sfrs, log_balmer_ssfrs = perform_sfr_computation(halpha_lums, balmer_ahalphas, log_masses, imf=imf)
        balmer_sfrs = 10**log_balmer_sfrs
        group_df['recomputed_balmerdec_sfr'] = balmer_sfrs

        avs_balmer_from_stellar = correct_av_for_dust(AVs, factor=2)
        avs_balmer_6565 = apply_dust_law(avs_balmer_from_stellar, target_wave=6565)
        log_av_sfrs, log_av_ssfrs = perform_sfr_computation(halpha_lums, avs_balmer_6565, log_masses, imf=imf)
        av_sfrs = 10**log_av_sfrs
        group_df['recomputed_AV_sfr'] = av_sfrs

        group_df['recomputed_sfr'] = np.ones(len(log_av_sfrs))
        # Find where the Hbeta readings are good, use sfr_corr for that
        halpha_good = group_df['ha_detflag_sfr'] == 0
        hbeta_good = group_df['hb_detflag_sfr'] == 0
        use_balmer_dec = np.logical_and(halpha_good, hbeta_good)
        group_df.loc[use_balmer_dec, 'recomputed_sfr'] = group_df[use_balmer_dec]['recomputed_balmerdec_sfr']


        # Find where Halpha is good but Hbeta is not, and use the halpha sfrs for that
        use_av_sfr = ~use_balmer_dec
        group_df.loc[use_av_sfr, 'recomputed_sfr'] = group_df[use_av_sfr]['recomputed_AV_sfr']
        group_df.to_csv(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv', index=False)



def perform_sfr_computation(halpha_lums, balmer_ahalphas, log_median_masses, imf='subsolar', replace_nan=False):
    intrinsic_halpha_lums = correct_lum_for_dust(halpha_lums, balmer_ahalphas)
    halpha_sfrs = ha_lum_to_sfr(intrinsic_halpha_lums, imf='subsolar')
    log_halpha_sfrs = np.log10(halpha_sfrs)
     # Divide by mean mass for sSFR
    halpha_ssfrs = halpha_sfrs / (10**log_median_masses)
    log_halpha_ssfrs = np.log10(halpha_ssfrs)
    if replace_nan == True:
        log_halpha_sfrs = np.nan_to_num(log_halpha_sfrs, nan=-99)
        log_halpha_ssfrs = np.nan_to_num(log_halpha_ssfrs, nan=-99)

    return log_halpha_sfrs, log_halpha_ssfrs


def get_montecarlo_errs(log_median_masses, log_halpha_sfrs, log_halpha_ssfrs, median_redshifts, luminosity, imf='subsolar'):
    #save distriubiton of generated sfrs and ssfrs
    err_log_sfr_lows = []
    err_log_sfr_highs = []
    err_log_ssfr_lows = []
    err_log_ssfr_highs = []
    for groupID in range(len(log_median_masses)):
        monte_carlo_df = ascii.read(imd.emission_fit_dir + f'/emission_fit_monte_carlos/{groupID}_monte_carlo.csv').to_pandas()
        if luminosity == True:
            new_ha_lums = monte_carlo_df['ha_flux']
        else:
            new_ha_fluxes = monte_carlo_df['ha_flux']
            new_ha_lums = flux_to_luminosity(new_ha_fluxes, median_redshifts[groupID]) 
        new_balmer_decs = monte_carlo_df['balmer_dec']
        new_balmer_avs = compute_balmer_av(new_balmer_decs)
        new_balmer_ahalphas = compute_balmer_ahalpha_from_AV(new_balmer_avs, law='Cardelli')
        
        sfr_outs = perform_sfr_computation(new_ha_lums, new_balmer_ahalphas, log_median_masses.iloc[groupID])
        all_log_sfrs = sfr_outs[0] 
        all_log_ssfrs = sfr_outs[1]
        all_sfrs  = 10**all_log_sfrs
        all_ssfrs  = 10**all_log_ssfrs
        sfr_measured = 10**log_halpha_sfrs[groupID]
        ssfr_measured = 10**log_halpha_ssfrs[groupID]
        err_log_sfr_low = 0.4343 * ((sfr_measured - np.percentile(all_sfrs, 16)) / sfr_measured)
        err_log_sfr_high = 0.4343 * ((np.percentile(all_sfrs, 86) - sfr_measured) / sfr_measured)
        err_log_ssfr_low = 0.4343 * ((ssfr_measured - np.percentile(all_ssfrs, 16)) / ssfr_measured)
        err_log_ssfr_high = 0.4343 * ((np.percentile(all_ssfrs, 86) - ssfr_measured) / ssfr_measured)
        err_log_sfr_lows.append(err_log_sfr_low)
        err_log_sfr_highs.append(err_log_sfr_high)
        err_log_ssfr_lows.append(err_log_ssfr_low)
        err_log_ssfr_highs.append(err_log_ssfr_high)
        neb_sfr_err_df = pd.DataFrame(zip(all_sfrs, new_balmer_avs), columns = ['sfr', 'balmer_av'])
        imd.check_and_make_dir(imd.cluster_dir + '/sfr_errs/')
        neb_sfr_err_df.to_csv(imd.cluster_dir + f'/sfr_errs/{groupID}_sfr_errs.csv', index=False)
        
    return err_log_sfr_lows, err_log_sfr_highs, err_log_ssfr_lows, err_log_ssfr_highs



def get_sfr_errs(bootstrap, halpha_lums, err_halpha_lums, balmer_ahalphas, err_balmer_halphas_low, err_balmer_halphas_high, log_median_masses, log_halpha_sfrs, log_halpha_ssfrs, median_redshifts, luminosity, imf='subsolar'):
    #save distriubiton of generated sfrs and ssfrs
    err_sfr_lows = []
    err_sfr_highs = []
    err_ssfr_lows = []
    err_ssfr_highs = []
    for groupID in range(len(halpha_lums)):
        boot_dfs = [ascii.read(imd.emission_fit_dir + f'/emission_fitting_boot_csvs/{groupID}_emission_fits_{bootstrap_num}.csv').to_pandas() for bootstrap_num in range(bootstrap)]
        ha_row = boot_dfs[0][boot_dfs[0]['line_name'] == 'Halpha'].index[0]
        if luminosity == True:
            new_ha_lums = [boot_dfs[i]['flux'].iloc[ha_row] for i in range(bootstrap)]
        else:
            new_ha_fluxes = [boot_dfs[i]['flux'].iloc[ha_row] for i in range(bootstrap)]
            new_halpha_lums = [flux_to_luminosity(new_ha_fluxes[i], median_redshifts[groupID]) for i in range(bootstrap)]
        new_balmer_decs = [boot_dfs[i]['balmer_dec'].iloc[ha_row] for i in range(bootstrap)]
        new_balmer_avs = [compute_balmer_av(new_balmer_decs[i]) for i in range(bootstrap)]
        new_balmer_ahalphas = [compute_balmer_ahalpha_from_AV(new_balmer_avs[i], law='Cardelli') for i in range(bootstrap)]

        sfr_outs = [perform_sfr_computation(np.array(new_ha_lums[j1]), np.array(new_balmer_ahalphas[j1]), log_median_masses.iloc[groupID], replace_nan=True) for j1 in range(len(new_balmer_ahalphas))]
        all_log_sfrs = [sfr_outs[i][0] for i in range(len(sfr_outs))]
        all_log_ssfrs = [sfr_outs[i][1] for i in range(len(sfr_outs))]
        all_sfrs  = 10**np.array(all_log_sfrs)
        all_ssfrs  = 10**np.array(all_log_ssfrs)
        sfr_measured = 10**log_halpha_sfrs[groupID]
        ssfr_measured = 10**log_halpha_ssfrs[groupID]
        err_sfr_low = 0.4343 * ((sfr_measured - np.percentile(all_sfrs, 16, axis=0)) / sfr_measured)
        err_sfr_high = 0.4343 * ((np.percentile(all_sfrs, 86, axis=0) - sfr_measured) / sfr_measured)
        err_ssfr_low = 0.4343 * ((ssfr_measured - np.percentile(all_ssfrs, 16, axis=0)) / ssfr_measured)
        err_ssfr_high = 0.4343 * ((np.percentile(all_ssfrs, 86, axis=0) - ssfr_measured) / ssfr_measured)
        err_sfr_lows.append(err_sfr_low)
        err_sfr_highs.append(err_sfr_high)
        err_ssfr_lows.append(err_ssfr_low)
        err_ssfr_highs.append(err_ssfr_high)
        neb_sfr_err_df = pd.DataFrame(zip(all_sfrs, new_balmer_avs), columns = ['sfr', 'balmer_av'])
        imd.check_and_make_dir(imd.cluster_dir + '/sfr_errs/')
        neb_sfr_err_df.to_csv(imd.cluster_dir + f'/sfr_errs/{groupID}_sfr_errs.csv', index=False)
    return err_sfr_lows, err_sfr_highs, err_ssfr_lows, err_ssfr_highs

def draw_asymettric_error(center, low_err, high_err):
    """Draws a point from two asymmetric normal distributions"""
    x = random.uniform(0,1)
    if x < 0.5:
        draw = np.random.normal(loc=0, scale=low_err)
        new_value = center - np.abs(draw)
    else:
        draw = np.random.normal(loc=0, scale=high_err)
        new_value = center + np.abs(draw)
    return new_value

def compute_balmer_ahalpha_from_AV(balmer_avs, law='Cardelli'):
    """Compues the Balmer Halpha given the AV"""
    if law=='Calzetti':
        balmer_halphas = 3.33*(balmer_avs / 4.05)
    if law=='Cardelli':
        balmer_halphas = balmer_avs * cardelli_law(6563)
    return balmer_halphas

def compute_balmer_AV_from_ahalpha(balmer_halphas, law='Cardelli'):
    """Compues the Balmer Halpha given the AV"""
    if law=='Cardelli':
        balmer_avs = balmer_halphas * cardelli_law(6563)
    return balmer_avs

def cardelli_law(wavelength_ang):
    #valid for 0.3um < wave < 1.1um
    wavelength_um = wavelength_ang/10000
    wavelength_inv_um = 1/wavelength_um
    if wavelength_inv_um > 0.3 and wavelength_inv_um < 1.1:
        a_x = 0.574*wavelength_inv_um**(1.61)
        b_x = -0.527*wavelength_inv_um**(1.61)
    elif wavelength_inv_um >= 1.1 and wavelength_inv_um < 3.3:
        y = wavelength_inv_um - 1.82
        a_x = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b_x = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    A_lambda_divide_AV = (a_x+b_x/3.1)
    return A_lambda_divide_AV

# compute_cluster_sfrs(luminosity=True, bootstrap=1000)