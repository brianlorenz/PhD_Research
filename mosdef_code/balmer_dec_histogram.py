# Plot a histogram of all the bootstrapped balmer decrements for each group
import initialize_mosdef_dirs as imd
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt


def plot_balmer_hist(n_clusters, n_boots):
    for groupID in range(n_clusters):
        balmer_decs = []
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
        imd.check_and_make_dir(imd.cluster_dir+'/cluster_stats/balmer_decs')
        fig.savefig(imd.cluster_dir+f'/cluster_stats/balmer_decs/{groupID}_balmer_hist.pdf')

# plot_balmer_hist(19, 100)