from astropy.io import ascii
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
import numpy as np

def compute_balmer_av(balmer_dec):
            balmer_av = 4.05*1.97*np.log10(balmer_dec/2.86)
            return balmer_av

def plot_balmer_stellar_avs(save_name):
    '''Makes a series of plots involving the balmer and stellar avs
    
    Parameters:
    save_name (str): Folder where everything is located
    '''

    # Make an output folder
    imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/av_plots/')

    fontsize = 14

    # Read in summary df
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()

    summary_df['balmer_stellar_av_ratio'] = summary_df['balmer_av'] / summary_df['av_median']
    summary_df['err_balmer_stellar_av_ratio_low'] = np.sqrt((summary_df['err_balmer_av_low']/summary_df['balmer_av'])**2 + (summary_df['err_av_median']/summary_df['av_median'])**2)
    summary_df['err_balmer_stellar_av_ratio_high'] = np.sqrt((summary_df['err_balmer_av_high']/summary_df['balmer_av'])**2 + (summary_df['err_av_median']/summary_df['av_median'])**2)

    # Fig 1: Balmer vs Stellar Avs
    fig, ax = plt.subplots(figsize=(8,8))    
    ax.errorbar(summary_df['av_median'], summary_df['balmer_av'], xerr=summary_df['err_av_median'], yerr=np.array(summary_df['err_balmer_av_low'], summary_df['err_balmer_av_high']), marker='o', ls='None', color='black')
    ax.set_xlabel('FAST $A_V$', fontsize=fontsize)
    ax.set_ylabel('Balmer $A_V$', fontsize=fontsize)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/av_plots/balmer_vs_stellar.pdf')

    # Fig 2: Balmer/Stellar ratio vs Properties
    def plot_prop(col_name, xlabel, plot_name):
        """Makes a plot of the balmer/stellar ratio vs another galaxy property

        Parameters:
        col_name (str): Name of the column to plot
        xlabel (str): Axis label for the x-axis
        plot_name (str): Sort name to append ot end of plot when saving
        """
        fig, ax = plt.subplots(figsize=(8,8))    
        ax.errorbar(summary_df[col_name], summary_df['balmer_stellar_av_ratio'], yerr=np.array(summary_df['err_balmer_stellar_av_ratio_low'], summary_df['err_balmer_stellar_av_ratio_high']), marker='o', ls='None', color='black')
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel('Balmer $A_V$ / FAST $A_V$', fontsize=fontsize)
        fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/av_plots/ratio_vs_{plot_name}.pdf')
    
    plot_prop('log_mass_median', 'Stellar Mass', 'mass')
    plot_prop('log_use_ssfr_median', 'SSFR', 'use_ssfr')
    plot_prop('log_use_sfr_median', 'SFR', 'use_sfr')



# plot_balmer_stellar_avs('both_4bin_1axis_median_params')


