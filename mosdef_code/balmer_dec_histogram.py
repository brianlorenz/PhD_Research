# Plot a histogram of all the bootstrapped balmer decrements for each group
import initialize_mosdef_dirs as imd
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from balmer_avs import compute_balmer_av
from compute_cluster_sfrs import compute_cluster_sfrs
from fit_emission import compute_err_and_logerr


def plot_balmer_hist(n_clusters, n_boots, skip_plots=False):
    groupIDs = []
    measured_balmers = []
    low_sigs = []
    low_2sigs = []
    high_sigs = []
    hbeta_two_sig_highs = []
    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    for groupID in range(n_clusters):
        print(f'Making balmer_dist_df group {groupID}')
        balmer_decs = []
        hbetas = []
        measured_balmer = cluster_summary_df[cluster_summary_df['groupID']==groupID]['balmer_dec'].iloc[0]
        for i in range(n_boots):
            emission_df = ascii.read(imd.cluster_dir+f'/emission_fitting/emission_fitting_boot_csvs/{groupID}_emission_fits_{i}.csv').to_pandas()
            balmer_decs.append(emission_df['balmer_dec'].iloc[0])
            hb_row = emission_df['line_name'] == 'Hbeta'
            hbetas.append(emission_df[hb_row]['flux'].iloc[0])


        min_balmer = np.floor(np.percentile(balmer_decs,0))
        max_balmer = np.ceil(np.percentile(balmer_decs, 100))
        if max_balmer > 30:
            max_balmer=30
        bins = np.arange(min_balmer, max_balmer, 0.1)

        one_sig_low = np.percentile(balmer_decs, 16)
        one_sig_high = np.percentile(balmer_decs, 84)
        two_sig_low = np.percentile(balmer_decs, 2.5)
        two_sig_high = np.percentile(balmer_decs, 97.5)
        hbeta_two_sig_high = np.percentile(hbetas, 97.5)

       
        groupIDs.append(groupID)
        measured_balmers.append(measured_balmer)
        low_sigs.append(one_sig_low)
        low_2sigs.append(two_sig_low)
        high_sigs.append(one_sig_high)
        hbeta_two_sig_highs.append(hbeta_two_sig_high)

        if skip_plots == False:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.hist(balmer_decs, bins=bins)
            def plot_1sig(sig_val, color='red'):
                ax.axvline(sig_val, color=color, ls='--')
            plot_1sig(one_sig_low)
            plot_1sig(one_sig_high)
            plot_1sig(two_sig_high, color='orange')
            plot_1sig(two_sig_low, color='orange')
        

            xlims = ax.get_xlim()
            if xlims[1]>30:
                ax.set_xlim(-1, 40)
            
            imd.check_and_make_dir(imd.cluster_dir+'/cluster_stats/balmer_decs')
            fig.savefig(imd.cluster_dir+f'/cluster_stats/balmer_decs/{groupID}_balmer_hist.pdf')
    balmer_dist_df = pd.DataFrame(zip(groupIDs, measured_balmers, low_sigs, high_sigs, low_2sigs, hbeta_two_sig_highs), columns=['groupID', 'balmer_dec', 'one_sig_balmer_low', 'one_sig_balmer_high', 'two_sig_balmer_low', 'two_sig_hbeta_high'])
    balmer_dist_df.to_csv(imd.cluster_dir+'/balmer_dist_df.csv', index=False)

def compute_balmer_lower_limits(sig_noise_thresh=3, hb_sig_noise_thresh=2):
    """Finds which groups should be assigned lower limits, and at what value

    sig_noise_thresh (float): balmer decrement value ratio compared to the width of the two-sigma distribution
    
    """
    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    balmer_dist_df = ascii.read(imd.cluster_dir+'/balmer_dist_df.csv').to_pandas()
    balmer_snr = cluster_summary_df['balmer_dec_snr']
    lower_limit_flag = balmer_snr<sig_noise_thresh
    lower_limit_flag_binary = 1*(lower_limit_flag)
    cluster_summary_df['flag_balmer_lower_limit'] = lower_limit_flag_binary

    cluster_summary_df['hb_snr'] = cluster_summary_df['hb_flux'] / cluster_summary_df['err_hb_flux']
    hb_limit_flag = cluster_summary_df['hb_snr']<hb_sig_noise_thresh
    hb_limit_flag_binary = 1*(hb_limit_flag)
    cluster_summary_df['flag_hb_limit'] = hb_limit_flag_binary

    cluster_summary_df['flag_balmer_lower_limit'] = cluster_summary_df['flag_hb_limit']

    #Measure what the limit should be
    # confident_balmer_decs = cluster_summary_df[cluster_summary_df['flag_balmer_lower_limit']==0]['balmer_dec'] 
    # balmer_limit = np.max(confident_balmer_decs)
    cluster_summary_df['balmer_dec_with_limit'] = cluster_summary_df['balmer_dec']
    cluster_summary_df['err_balmer_dec_with_limit_low'] = cluster_summary_df['err_balmer_dec_low']
    cluster_summary_df['err_balmer_dec_with_limit_high'] = cluster_summary_df['err_balmer_dec_high']
    cluster_summary_df['err_balmer_av_with_limit_low'] = cluster_summary_df['err_balmer_av_low']
    cluster_summary_df['err_balmer_av_with_limit_high'] = cluster_summary_df['err_balmer_av_high']
    cluster_summary_df['hb_flux_upper_limit'] = cluster_summary_df['hb_flux']
    cluster_summary_df['log_O3_Hb_lower_limit'] = cluster_summary_df['log_O3_Hb']
    cluster_summary_df['O3N2_metallicity_upper_limit'] = cluster_summary_df['O3N2_metallicity']
    for i in range(len(cluster_summary_df)):
        if cluster_summary_df.iloc[i]['flag_balmer_lower_limit'] == 1:
            balmer_limit = balmer_dist_df.iloc[i]['two_sig_balmer_low']
            #Compute using AV of 0 (dec = 2.86) if limit is less than 2.86
            if balmer_limit < 2.86:
                balmer_limit = 2.86
            cluster_summary_df.loc[i, 'balmer_dec_with_limit'] = balmer_limit
            cluster_summary_df.loc[i, 'err_balmer_dec_with_limit_low'] = 0
            cluster_summary_df.loc[i, 'err_balmer_dec_with_limit_high'] = 0
            cluster_summary_df.loc[i, 'err_balmer_av_with_limit_low'] = 0
            cluster_summary_df.loc[i, 'err_balmer_av_with_limit_high'] = 0
        if cluster_summary_df.iloc[i]['flag_hb_limit'] == 1:
            emission_df = ascii.read(imd.cluster_dir+f'/emission_fitting/emission_fitting_csvs/{i}_emission_fits.csv').to_pandas()
            hb_limit = balmer_dist_df.iloc[i]['two_sig_hbeta_high']
            oiii_row = emission_df['line_name'] == 'O3_5008'
            oiii_flux = emission_df[oiii_row]['flux'].iloc[0]
            ha_row = emission_df['line_name'] == 'Halpha'
            ha_flux = emission_df[ha_row]['flux'].iloc[0]
            nii_row = emission_df['line_name'] == 'N2_6585'
            nii_flux = emission_df[nii_row]['flux'].iloc[0]
            cluster_summary_df.loc[i, 'hb_flux_upper_limit'] = hb_limit
            O3N2_numerator, log_O3_hb_lim, _, _ = compute_err_and_logerr(oiii_flux, hb_limit, -99, -99)
            cluster_summary_df.loc[i, 'log_O3_Hb_lower_limit'] = log_O3_hb_lim
            O3N2_denominator, _, _, _ = compute_err_and_logerr(nii_flux, ha_flux, -99, -99)
            _, O3N2, _, _ = compute_err_and_logerr(O3N2_numerator, O3N2_denominator, -99, -99)
            metal_limit = 8.97-0.39*O3N2
            cluster_summary_df.loc[i, 'O3N2_metallicity_upper_limit'] = metal_limit

    #Compute the balmer avs using these limits
    cluster_summary_df['balmer_av_with_limit'] = compute_balmer_av(cluster_summary_df['balmer_dec_with_limit'])
    
    # Save the data frame with the limit added
    cluster_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)

    compute_cluster_sfrs(lower_limit=True)
    


# plot_balmer_hist(20, 1000, skip_plots=True)
# compute_balmer_lower_limits()