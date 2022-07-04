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
    label_font = 18

    # fig = plt.figure(figsize=(17, 8))
    # ax_av = fig.add_axes([0.01, 0.2, 0.45, 0.6])
    # ax_beta = fig.add_axes([0.42, 0.2, 0.45, 0.6])
    # ax_cbar = fig.add_axes([0.90, 0.2, 0.02, 0.60])
    fig, axarr = plt.subplots(1, 2, figsize=(21, 8))
    ax_av = axarr[0]
    ax_beta = axarr[1]
    plot_balmer_dec(save_name, nbins, split_by, y_var='av', color_var=split_by, mass_ax=ax_av, fig=fig)
    plot_balmer_dec(save_name, nbins, split_by, y_var='beta', color_var=split_by, mass_ax=ax_beta, fig=fig)
    for ax in [ax_av, ax_beta]:
        ax.tick_params(labelsize=18)
        ax.set_xlabel(stellar_mass_label, fontsize=label_font)
        scale_aspect(ax)
    ax_av.set_ylabel('A$_V$', fontsize=label_font)
    ax_beta.set_ylabel('$\\beta$', fontsize=label_font)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/av_beta_combined.pdf',bbox_inches='tight')

def plot_mass_met_paper(nbins, split_by, save_name):
    label_font = 18

    # fig = plt.figure(figsize=(17, 8))
    # ax_av = fig.add_axes([0.01, 0.2, 0.45, 0.6])
    # ax_beta = fig.add_axes([0.42, 0.2, 0.45, 0.6])
    # ax_cbar = fig.add_axes([0.90, 0.2, 0.02, 0.60])
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    plot_balmer_dec(save_name, nbins, split_by, y_var='metallicity', color_var=split_by, mass_ax=ax, fig=fig)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(stellar_mass_label, fontsize=label_font)
    scale_aspect(ax)
    ax.set_ylabel(metallicity_label, fontsize=label_font)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/mass_metallicity.pdf',bbox_inches='tight')

# plot_AV_beta_paper(8, 'log_use_sfr', 'whitaker_sfms_boot100')
# plot_mass_met_paper(8, 'log_use_sfr', 'whitaker_sfms_boot100')