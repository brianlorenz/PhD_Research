# Plot a histogram of all the bootstrapped balmer decrements for each group
import initialize_mosdef_dirs as imd
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from balmer_avs import compute_balmer_av
from compute_cluster_sfrs import compute_cluster_sfrs


def plot_balmer_hist(n_clusters, n_boots):
    groupIDs = []
    measured_balmers = []
    low_sigs = []
    low_2sigs = []
    high_sigs = []
    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    for groupID in range(n_clusters):
        balmer_decs = []
        measured_balmer = cluster_summary_df[cluster_summary_df['groupID']==groupID]['balmer_dec'].iloc[0]
        for i in range(n_boots):
            emission_df = ascii.read(imd.cluster_dir+f'/emission_fitting/emission_fitting_boot_csvs/{groupID}_emission_fits_{i}.csv').to_pandas()
            balmer_decs.append(emission_df['balmer_dec'].iloc[0])
        


        min_balmer = np.floor(np.percentile(balmer_decs,0))
        max_balmer = np.ceil(np.percentile(balmer_decs, 100))
        if max_balmer > 30:
            max_balmer=30
        bins = np.arange(min_balmer, max_balmer, 0.1)

        one_sig_low = np.percentile(balmer_decs, 16)
        one_sig_high = np.percentile(balmer_decs, 84)
        two_sig_low = np.percentile(balmer_decs, 2.5)
        two_sig_high = np.percentile(balmer_decs, 97.5)

        fig, ax = plt.subplots(figsize=(8,8))
        ax.hist(balmer_decs, bins=bins)
        def plot_1sig(sig_val, color='red'):
            ax.axvline(sig_val, color=color, ls='--')
        plot_1sig(one_sig_low)
        plot_1sig(one_sig_high)
        plot_1sig(two_sig_high, color='orange')
        plot_1sig(two_sig_low, color='orange')
       
        groupIDs.append(groupID)
        measured_balmers.append(measured_balmer)
        low_sigs.append(one_sig_low)
        low_2sigs.append(two_sig_low)
        high_sigs.append(one_sig_high)

        xlims = ax.get_xlim()
        if xlims[1]>30:
            ax.set_xlim(-1, 40)
        
        imd.check_and_make_dir(imd.cluster_dir+'/cluster_stats/balmer_decs')
        fig.savefig(imd.cluster_dir+f'/cluster_stats/balmer_decs/{groupID}_balmer_hist.pdf')
    balmer_dist_df = pd.DataFrame(zip(groupIDs, measured_balmers, low_sigs, high_sigs, low_2sigs), columns=['groupID', 'balmer_dec', 'one_sig_balmer_low', 'one_sig_balmer_high', 'two_sig_balmer_low'])
    balmer_dist_df.to_csv(imd.cluster_dir+'/balmer_dist_df.csv', index=False)

def compute_balmer_lower_limits(sig_noise_thresh=3):
    """Finds which groups should be assigned lower limits, and at what value

    sig_noise_thresh (float): balmer decrement value ratio compared to the width of the two-sigma distribution
    
    """
    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    balmer_dist_df = ascii.read(imd.cluster_dir+'/balmer_dist_df.csv').to_pandas()
    balmer_snr = cluster_summary_df['balmer_dec_snr']
    lower_limit_flag = balmer_snr<sig_noise_thresh
    lower_limit_flag_binary = 1*(lower_limit_flag)
    cluster_summary_df['flag_balmer_lower_limit'] = lower_limit_flag_binary

    #Measure what the limit should be
    # confident_balmer_decs = cluster_summary_df[cluster_summary_df['flag_balmer_lower_limit']==0]['balmer_dec'] 
    # balmer_limit = np.max(confident_balmer_decs)
    cluster_summary_df['balmer_dec_with_limit'] = cluster_summary_df['balmer_dec']
    for i in range(len(cluster_summary_df)):
        if cluster_summary_df.iloc[i]['flag_balmer_lower_limit'] == 1:
            balmer_limit = balmer_dist_df.iloc[i]['two_sig_balmer_low']
            #Compute using AV of 0 (dec = 2.86) if limit is less than 2.86
            if balmer_limit < 2.86:
                balmer_limit = 2.86
            cluster_summary_df.loc[i, 'balmer_dec_with_limit'] = balmer_limit
    #Compute the balmer avs using these limits
    cluster_summary_df['balmer_av_with_limit'] = compute_balmer_av(cluster_summary_df['balmer_dec_with_limit'])
    
    # Save the data frame with the limit added
    cluster_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)

    # compute_cluster_sfrs(lower_limit=True)
    


# plot_balmer_hist(19, 1000)
# compute_balmer_lower_limits()