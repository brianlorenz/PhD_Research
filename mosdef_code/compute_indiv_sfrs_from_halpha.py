import initialize_mosdef_dirs as imd
from cosmology_calcs import flux_to_luminosity
from compute_new_sfrs import correct_lum_for_dust, ha_lum_to_sfr
import numpy as np
from astropy.io import ascii

# CURRENTLY IGNORES NANS WHEN COMPUTING - SOME OF THE GALAXIES DON'T HAVE HA DETECTIONS
def compute_indiv_sfrs(n_clusters, lower_limit=True):
    """
    
    Parameters:
    lower_limit (boolean): Set to true to use the lower limits computed in balmer_dec_histogram
    luminosity (boolean): Set to false if the fluxes are already in luminosity space
    """
    cluster_summary_df = imd.read_cluster_summary_df()
    median_sfrs = []
    median_ssfrs = []

    for groupID in range(n_clusters):
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        halpha_fluxes = group_df['ha_flux']
        cluster_summary_df.iloc[groupID]['balmer_av']
        if lower_limit == True:
            balmer_av = cluster_summary_df.iloc[groupID]['balmer_av_with_limit']
        redshift = group_df['Z_MOSFIRE']
        log_mass = group_df['log_mass']
   
   

        # Another method is to try to compute using A_V rather than mass
        AV = cluster_summary_df.iloc[groupID]['AV']

        #Convert the Balmer AV to A_Halpha using https://iopscience.iop.org/article/10.1088/0004-637X/763/2/145/pdf
        balmer_ahalpha = 3.33*(balmer_av / 4.05)


        # Convert ha to luminsoty
        halpha_lums = flux_to_luminosity(halpha_fluxes, redshift)
    

        # Get dust-corrected halpha
        intrinsic_halpha_lums = correct_lum_for_dust(halpha_lums, balmer_ahalpha) 
        # intrinsic_halpha_lums = correct_ha_lum_for_dust(halpha_lums, ahalphas)

        # Derive SFR from Hao 2011
        halpha_sfrs = ha_lum_to_sfr(intrinsic_halpha_lums, imf='Hao_Chabrier')
        log_halpha_sfrs = np.log10(halpha_sfrs)

        # Divide by mean mass for sSFR
        halpha_ssfrs = halpha_sfrs / (10**log_mass)
        log_halpha_ssfrs = np.log10(halpha_ssfrs)

        group_df['computed_log_sfr'] = log_halpha_sfrs
        group_df['computed_log_ssfr'] = log_halpha_ssfrs
        group_df['computed_log_sfr_with_limit'] = -99
        group_df['computed_log_ssfr_with_limit'] = -99
        
        if lower_limit==True:
            group_df['computed_log_sfr'] = -99
            group_df['computed_log_ssfr'] = -99
            group_df['computed_log_sfr_with_limit'] = log_halpha_sfrs
            group_df['computed_log_ssfr_with_limit'] = log_halpha_ssfrs
        


        median_sfr = log_halpha_sfrs.median()
        median_ssfr = log_halpha_ssfrs.median()
        median_sfrs.append(median_sfr)
        median_ssfrs.append(median_ssfr)
    
        group_df.to_csv(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv', index=False)
    
    if lower_limit==True:
        cluster_summary_df['median_indiv_computed_log_sfr'] = -99
        cluster_summary_df['median_indiv_computed_log_ssfr'] = -99
        cluster_summary_df['median_indiv_computed_log_sfr_with_limit'] = median_sfrs
        cluster_summary_df['median_indiv_computed_log_ssfr_with_limit'] = median_ssfrs
    else:
        cluster_summary_df['median_indiv_computed_log_sfr'] = median_sfrs
        cluster_summary_df['median_indiv_computed_log_ssfr'] = median_ssfrs
        cluster_summary_df['median_indiv_computed_log_sfr_with_limit'] = -99
        cluster_summary_df['median_indiv_computed_log_ssfr_with_limit'] = -99
    cluster_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)



# compute_indiv_sfrs(19)