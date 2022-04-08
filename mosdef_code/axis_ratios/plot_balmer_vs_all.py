# Plots the balmer decrement vs a vairiety of properies
from tkinter import font
import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np

def plot_balmer_vs_all(save_name):
    """Plots balmer decrement vs a variety of measured galaxy propertties
    
    Parameters:
    save_name (str): Directory to save the images
    
    """
    
    # Make an output folder
    imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/')

    fontsize = 14
    
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()


    fig, axarr = plt.subplots(3,3,figsize = (20,20))

    colors = ['black', 'blue', 'orange', 'mediumseagreen', 'red', 'violet', 'grey', 'pink', 'cyan', 'darkblue', 'brown', 'darkgreen']

    def plot_balmer_on_axis(ax, x_points, err_x_points='None'):
        """Makes one of the plots
        
        Parameters:
        ax (matplotlib.axes): Axis to plot on
        x_points (str): x_value column name to plot
        err_x_points (str): uncertainties on x_values to plot column name, can be 'None'
        """
        for i in range(len(summary_df)):
            row = summary_df.iloc[i]
            if err_x_points=='None':
                xerr=None
            else:
                xerr=row[err_x_points]
            ax.errorbar(row[x_points], row['balmer_dec'], yerr=np.array(row['err_balmer_dec_low'], row['err_balmer_dec_high']), xerr=xerr, color=colors[i], marker='o', ls='None')
            ax.set_xlabel(x_points, fontsize=fontsize)
            ax.set_ylabel('Balmer Decrement', fontsize=fontsize)
            ax.set_ylim(2, 7)
    

    plot_balmer_on_axis(axarr[0,0], 'log_mass_median')
    plot_balmer_on_axis(axarr[0,1], 'metallicity_median', 'err_metallicity_median')
    plot_balmer_on_axis(axarr[0,2], 'log_use_sfr_median')
    plot_balmer_on_axis(axarr[1,2], 'log_use_ssfr_median')
    plot_balmer_on_axis(axarr[1,0], 'av_median')
    plot_balmer_on_axis(axarr[1,1], 'beta_median')



    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/balmer_plots.pdf')


plot_balmer_vs_all('both_ssfrs_4bin_median_2axis')