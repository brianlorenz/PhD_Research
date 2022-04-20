# Makes a plot of the overview with the Av, Beta, Balmer, Metallicity, Sample
from plot_sample_split import plot_sample_split
from plot_balmer_dec import plot_balmer_dec
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd

def plot_overview(nbins, save_name, ratio_bins, starting_points, mass_width, split_width, sfms_bins, split_by):
    fig, axarr = plt.subplots(2, 5, figsize=(20, 10))
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
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/combined_overview.pdf')