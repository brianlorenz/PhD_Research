import initialize_mosdef_dirs as imd
from cosmology_calcs import flux_to_luminosity
from compute_new_sfrs import correct_ha_lum_for_dust, ha_lum_to_sfr
import numpy as np
from astropy.io import ascii

def compute_cluster_sfrs(lower_limit=True, luminosity=False, prospector=False):
    """
    
    Parameters:
    lower_limit (boolean): Set to true to use the lower limits computed in balmer_dec_histogram
    luminosity (boolean): Set to false if the fluxes are already in luminosity space
    prospector (boolean): Set to true if you want to compute the prospector SFRs
    """
    cluster_summary_df = imd.read_cluster_summary_df()

    halpha_fluxes = cluster_summary_df['ha_flux']
    balmer_avs = cluster_summary_df['balmer_av']
    balmer_decs = cluster_summary_df['balmer_dec']
    if lower_limit == True:
        balmer_avs = cluster_summary_df['balmer_av_with_limit']
        balmer_decs = cluster_summary_df['balmer_dec_with_limit']
   
    # log_mean_masses = cluster_summary_df['mean_log_mass']
    

    # For redshifts, we now want to use the target galaxy rather than the average of the group
    # redshifts = cluster_summary_df['redshift']
    group_dfs = [ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas() for groupID in range(len(cluster_summary_df))]
    norm_gals = [group_dfs[i][group_dfs[i]['norm_factor']==1] for i in range(len(cluster_summary_df))]
    redshifts = [norm_gals[i]['Z_MOSFIRE'].iloc[0] for i in range(len(cluster_summary_df))]
    cluster_summary_df['target_galaxy_redshifts'] = redshifts
    cluster_summary_df['target_galaxy_median_log_mass'] = [norm_gals[i]['log_mass'].iloc[0] for i in range(len(cluster_summary_df))]
    
    median_redshifts = cluster_summary_df['redshift']
    log_norm_median_masses = cluster_summary_df['norm_median_log_mass']
    log_median_masses = cluster_summary_df['median_log_mass']

    # Another method is to try to compute using A_V rather than mass
    AV = cluster_summary_df['AV']

    #Convert the Balmer AV to A_Halpha using https://iopscience.iop.org/article/10.1088/0004-637X/763/2/145/pdf
    balmer_ahalphas = 3.33*(balmer_avs / 4.05)

    # ahalphas = 3.33*(AV / 4.05)*2

    # Convert ha to luminsoty
    halpha_lums = flux_to_luminosity(halpha_fluxes, median_redshifts)
    if luminosity == True:
        halpha_lums = halpha_fluxes
    if prospector == True:
        halpha_lums = cluster_summary_df['prospector_halpha_luminosity']
        log_median_masses = np.log10(cluster_summary_df['surviving_mass50'])
        prospector_balmer_avs = cluster_summary_df['prospector_balmer_av']
        prospector_balmer_ahalphas = 3.33*(prospector_balmer_avs / 4.05)

    # Get dust-corrected halpha
    intrinsic_halpha_lums = correct_ha_lum_for_dust(halpha_lums, balmer_ahalphas) 
    if prospector == True:
        prospector_av_intrinsic_halpha_lums = correct_ha_lum_for_dust(halpha_lums, prospector_balmer_ahalphas) 
    # intrinsic_halpha_lums = correct_ha_lum_for_dust(halpha_lums, ahalphas)

    # Derive SFR from Hao 2011
    halpha_sfrs = ha_lum_to_sfr(intrinsic_halpha_lums, imf='Hao_Chabrier')
    if prospector == True:
        prospector_av_halpha_sfrs = ha_lum_to_sfr(prospector_av_intrinsic_halpha_lums, imf='Hao_Chabrier')
        prospector_av_log_halpha_sfrs = np.log10(prospector_av_halpha_sfrs)
        prospector_av_halpha_ssfrs = prospector_av_halpha_sfrs / (10**log_median_masses)
        prospector_av_log_halpha_ssfrs = np.log10(prospector_av_halpha_ssfrs)
    log_halpha_sfrs = np.log10(halpha_sfrs)

    # Divide by mean mass for sSFR
    halpha_ssfrs = halpha_sfrs / (10**log_median_masses)
    log_halpha_ssfrs = np.log10(halpha_ssfrs)

    if prospector == True:
        cluster_summary_df['prospector_log_sfr'] = prospector_av_log_halpha_sfrs
        cluster_summary_df['prospector_log_ssfr'] = prospector_av_log_halpha_ssfrs
        cluster_summary_df['cluster_av_prospector_sfr'] = log_halpha_sfrs
        cluster_summary_df['cluster_av_prospector_log_ssfr'] = log_halpha_ssfrs
    else:
        cluster_summary_df['computed_log_sfr'] = log_halpha_sfrs
        cluster_summary_df['computed_log_ssfr'] = log_halpha_ssfrs
        cluster_summary_df['computed_log_sfr_with_limit'] = -99
        cluster_summary_df['computed_log_ssfr_with_limit'] = -99
        if lower_limit==True:
            cluster_summary_df['computed_log_sfr'] = -99
            cluster_summary_df['computed_log_ssfr'] = -99
            cluster_summary_df['computed_log_sfr_with_limit'] = log_halpha_sfrs
            cluster_summary_df['computed_log_ssfr_with_limit'] = log_halpha_ssfrs

    cluster_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)



# compute_cluster_sfrs()