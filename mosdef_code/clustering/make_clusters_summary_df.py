# Makes a datafrmae with summarized properties from all the clusters
import initialize_mosdef_dirs as imd
import numpy as np
from astropy.io import ascii
import pandas as pd

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

    balmer_decs = []
    err_balmer_dec_lows = []
    err_balmer_dec_highs = []


    O3N2_metallicities = []
    err_O3N2_metallicity_lows = []
    err_O3N2_metallicity_highs = []

    

    
    for groupID in range(n_clusters):
        if groupID in ignore_groups:
            print(f'Ignoring group {groupID}')
            continue

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

        # Read in the emission fits:
        emission_df = ascii.read(imd.emission_fit_csvs_dir + f'/{groupID}_emission_fits.csv').to_pandas()
        balmer_decs.append(emission_df.iloc[0]['balmer_dec'])
        err_balmer_dec_lows.append(emission_df.iloc[0]['err_balmer_dec_low'])
        err_balmer_dec_highs.append(emission_df.iloc[0]['err_balmer_dec_high'])

        O3N2_metallicities.append(emission_df.iloc[0]['O3N2_metallicity'])
        err_O3N2_metallicity_lows.append(emission_df.iloc[0]['err_O3N2_metallicity_low'])
        err_O3N2_metallicity_highs.append(emission_df.iloc[0]['err_O3N2_metallicity_high'])

        # Read in metallicity fits
        metals_df = ascii.read(imd.cluster_dir + f'/cluster_metallicities.csv').to_pandas()

    # Build into DataFrame
    clusters_summary_df = pd.DataFrame(zip(groupIDs, n_galss, median_zs, median_masses, median_sfrs, median_ssfrs, median_vjs, median_uvs, balmer_decs, err_balmer_dec_lows, err_balmer_dec_highs, O3N2_metallicities, err_O3N2_metallicity_lows, err_O3N2_metallicity_highs), columns=['groupID', 'n_gals', 'redshift', 'log_mass', 'log_sfr', 'log_ssfr', 'median_V_J', 'median_U_V', 'balmer_dec', 'err_balmer_dec_low', 'err_balmer_dec_high', 'O3N2_metallicity', 'err_O3N2_metallicity_low', 'err_O3N2_metallicity_high'])
    clusters_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)

# make_clusters_summary_df(23)