# Plot a histogram of all the bootstrapped balmer decrements for each group
import initialize_mosdef_dirs as imd
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from balmer_avs import compute_balmer_av


def plot_balmer_hist(n_clusters, n_boots):
    groupIDs = []
    measured_balmers = []
    low_sigs = []
    high_sigs = []
    for groupID in range(n_clusters):
        balmer_decs = []
        cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
        measured_balmer = cluster_summary_df[cluster_summary_df['groupID']==groupID]['balmer_dec'].iloc[0]
        for i in range(n_boots):
            emission_df = ascii.read(imd.cluster_dir+f'/emission_fitting/emission_fitting_boot_csvs/{groupID}_emission_fits_{i}.csv').to_pandas()
            balmer_decs.append(emission_df['balmer_dec'].iloc[0])
        


        min_balmer = np.floor(np.percentile(balmer_decs,5))
        max_balmer = np.ceil(np.percentile(balmer_decs, 95))
        if max_balmer > 30:
            max_balmer=30
        bins = np.arange(min_balmer, max_balmer, 0.1)

        one_sig_low = np.percentile(balmer_decs, 16)
        one_sig_high = np.percentile(balmer_decs, 84)

        fig, ax = plt.subplots(figsize=(8,8))
        ax.hist(balmer_decs, bins=bins)
        def plot_1sig(sig_val):
            ax.axvline(sig_val, color='red', ls='--')
        plot_1sig(one_sig_low)
        plot_1sig(one_sig_high)
       
        groupIDs.append(groupID)
        measured_balmers.append(measured_balmer)
        low_sigs.append(one_sig_low)
        high_sigs.append(one_sig_high)
        
        imd.check_and_make_dir(imd.cluster_dir+'/cluster_stats/balmer_decs')
        fig.savefig(imd.cluster_dir+f'/cluster_stats/balmer_decs/{groupID}_balmer_hist.pdf')
    balmer_dist_df = pd.DataFrame(zip(groupIDs, measured_balmers, low_sigs, high_sigs), columns=['groupID', 'balmer_dec', 'one_sig_balmer_low', 'one_sig_balmer_high'])
    balmer_dist_df.to_csv(imd.cluster_dir+'/balmer_dist_df.csv')

def compute_balmer_lower_limits(std_flag_thresh=3):
    """Finds which groups should be assigned lower limits, and at what value

    std_flag_thresh (float): wide of a 84-16th percentile range to allower before flagging
    
    """
    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    balmer_std = ((cluster_summary_df['err_balmer_dec_high']+cluster_summary_df['balmer_dec'])-(cluster_summary_df['balmer_dec']-cluster_summary_df['err_balmer_dec_low']))
    lower_limit_flag = balmer_std>std_flag_thresh
    lower_limit_flag_binary = 1*(lower_limit_flag)
    cluster_summary_df['flag_balmer_lower_limit'] = lower_limit_flag_binary

    #Measure what the limit should be
    confident_balmer_decs = cluster_summary_df[cluster_summary_df['flag_balmer_lower_limit']==0]['balmer_dec'] 
    balmer_limit = np.max(confident_balmer_decs)
    cluster_summary_df['balmer_dec_with_limit'] = cluster_summary_df['balmer_dec']
    cluster_summary_df.loc[cluster_summary_df['flag_balmer_lower_limit']==1, 'balmer_dec_with_limit'] = balmer_limit
    #Compute the balmer avs using these limits
    cluster_summary_df['balmer_av_with_limit'] = compute_balmer_av(cluster_summary_df['balmer_dec_with_limit'])
    
    # Save the data frame with the limit added
    cluster_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)
    

# compute_balmer_lower_limits()
# plot_balmer_hist(19, 100)
# 2, 7, 13, 17