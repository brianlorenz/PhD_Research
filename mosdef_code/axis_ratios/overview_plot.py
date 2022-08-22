# Makes a plot of the overview with the Av, Beta, Balmer, Metallicity, Sample
from plot_sample_split import plot_sample_split
from plot_balmer_dec import plot_balmer_dec
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
import numpy as np
from plot_vals import *

def plot_overview(nbins, save_name, ratio_bins, starting_points, mass_width, split_width, sfms_bins, split_by):
    fig, axarr = plt.subplots(2, 5, figsize=(30, 10))
    ax_sample = axarr[0,0]
    ax_balmer1 = axarr[0,1]
    ax_balmer2 = axarr[0,2]
    axarr_balmer = [ax_balmer1, ax_balmer2]
    ax_av1 = axarr[0,3]
    ax_av2 = axarr[0,4]
    axarr_av = [ax_av1, ax_av2]
    ax_metals1 = axarr[1,1]
    ax_metals2 = axarr[1,2]
    axarr_metals = [ax_metals1, ax_metals2]
    ax_beta1 = axarr[1,3]
    ax_beta2 = axarr[1,4]
    axarr_beta = [ax_beta1, ax_beta2]
    
    
    plot_sample_split(nbins, save_name, ratio_bins, starting_points, mass_width, split_width, nbins, sfms_bins, ax=ax_sample, fig=fig)
    plot_balmer_dec(save_name, nbins, split_by, y_var='balmer_dec', color_var=split_by, axarr=axarr_balmer, fig=fig)
    plot_balmer_dec(save_name, nbins, split_by, y_var='metallicity', color_var=split_by, axarr=axarr_metals, fig=fig)
    plot_balmer_dec(save_name, nbins, split_by, y_var='av', color_var=split_by, axarr=axarr_av, fig=fig)
    plot_balmer_dec(save_name, nbins, split_by, y_var='beta', color_var=split_by, axarr=axarr_beta, fig=fig) 
    # plt.tight_layout()
    print('saving')
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/combined_overview.pdf')



def plot_AV_beta_paper(nbins, split_by, save_name):
    label_font = single_column_axisfont+4

    # fig = plt.figure(figsize=(17, 8))
    # ax_av = fig.add_axes([0.01, 0.2, 0.45, 0.6])
    # ax_beta = fig.add_axes([0.42, 0.2, 0.45, 0.6])
    # ax_cbar = fig.add_axes([0.90, 0.2, 0.02, 0.60])
    fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
    ax_av = axarr[0]
    ax_beta = axarr[1]
    plot_balmer_dec(save_name, nbins, split_by, y_var='av', color_var=split_by, mass_ax=ax_av, fig=fig)
    plot_balmer_dec(save_name, nbins, split_by, y_var='beta', color_var=split_by, mass_ax=ax_beta, fig=fig)
    for ax in [ax_av, ax_beta]:
        ax.tick_params(labelsize=label_font)
        ax.set_xlabel(stellar_mass_label, fontsize=label_font)
        scale_aspect(ax)
    ax_av.set_ylabel('A$_V$', fontsize=label_font)
    ax_beta.set_ylabel('$\\beta$', fontsize=label_font)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/av_beta_combined.pdf',bbox_inches='tight')

def plot_mass_met_paper(nbins, split_by, save_name):
    label_font = single_column_axisfont

    # fig = plt.figure(figsize=(17, 8))
    # ax_av = fig.add_axes([0.01, 0.2, 0.45, 0.6])
    # ax_beta = fig.add_axes([0.42, 0.2, 0.45, 0.6])
    # ax_cbar = fig.add_axes([0.90, 0.2, 0.02, 0.60])
    def sanders_metal_line(log_mass, gamma=0.34, z10=8.414):
        '''O3N2 Mass-metallicity relation from Sanders 2021
        
        '''
        m10 = log_mass-10
        metals = gamma*m10+z10
        return metals
    masses = np.arange(9, 11, 0.1)
    metals = sanders_metal_line(masses)

    sanders_high_sfr_mass = (9.56, 10.005, 10.29, 10.785)
    sanders_high_sfr_mass_err = (0.33, 0.105, 0.16, 0.325)
    sanders_high_sfr_metal = (8.13, 8.31, 8.32, 8.42)
    sanders_high_sfr_metal_err_high = (0.03, 0.03, 0.03, 0.03)
    sanders_high_sfr_metal_err_low = (0.05, 0.03, 0.04, 0.03)

    sanders_low_sfr_mass = (9.31, 9.635, 9.91, 10.235)
    sanders_low_sfr_mass_err = (0.16, 0.115, 0.11, 0.165)
    sanders_low_sfr_metal = (8.16, 8.16, 8.21, 8.38)
    sanders_low_sfr_metal_err_high = (0, 0.07, 0.04, 0.03)
    sanders_low_sfr_metal_err_low = (0.2, 0.09, 0.06, 0.04)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    plot_balmer_dec(save_name, nbins, split_by, y_var='metallicity', color_var=split_by, mass_ax=ax, fig=fig)
    
    #Add results from sanders papers?
    # ax.plot(masses, metals, marker='None', ls='--', label='Sanders+ 2021', color='grey', lw=3)
    # ax.errorbar(sanders_low_sfr_mass, sanders_low_sfr_metal, xerr=sanders_low_sfr_mass_err, yerr=np.array([sanders_low_sfr_metal_err_low, sanders_low_sfr_metal_err_high]), marker='o', label='Sanders+ (2015) Low SFR', color='blue', ls='None')
    # ax.errorbar(sanders_high_sfr_mass, sanders_high_sfr_metal, xerr=sanders_high_sfr_mass_err, yerr=np.array([sanders_high_sfr_metal_err_low, sanders_high_sfr_metal_err_high]), marker='o', label='Sanders+ (2015) High SFR', color='orange', ls='None')
    
    ax.tick_params(labelsize=label_font)
    ax.set_xlabel(stellar_mass_label, fontsize=label_font)
    scale_aspect(ax)
    # ax.legend(fontsize=16, loc=2)
    ax.set_ylabel(metallicity_label, fontsize=label_font)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/mass_metallicity.pdf',bbox_inches='tight')

# plot_AV_beta_paper(8, 'log_use_sfr', 'whitaker_sfms_boot100')
# plot_mass_met_paper(8, 'log_use_sfr', 'whitaker_sfms_boot100')