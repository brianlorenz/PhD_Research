import initialize_mosdef_dirs as imd
from overview_plot_of_clusters import make_overview_plot_clusters, setup_figs
from plot_cluster_a_vs_b import plot_a_vs_b_paper
import matplotlib.pyplot as plt
from plot_vals import *
from matplotlib.gridspec import GridSpec
from uvj_clusters import setup_uvj_plot
from astropy.io import ascii




def make_paper_plots(n_clusters, norm_method):
    # Overview figure
    # setup_figs(n_clusters, norm_method, bpt_color=True, paper_overview=False, prospector_spec=False)

    ### Potentially 4 panels? Or maybe different figures
    # Prospector AV vs Mass, and Balmer dec measured vs mass
    # AV vs Balmer decrement - how much extra attenuation?
    # Attenuation curve figure(s) - what controsl it

    # SFR comparison between prospector and emission lines


    #ssfr vs mass, metallicity vs mass
    def make_ssfr_mass_metallicity_fig():
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1],width_ratios=[1,1])
        ax_ssfr = fig.add_subplot(gs[0, 0])
        ax_metallicity = fig.add_subplot(gs[0, 1])
        plot_a_vs_b_paper('median_log_mass', 'computed_log_ssfr_with_limit', stellar_mass_label, ssfr_label, 'None', axis_obj=ax_ssfr, yerr=True, plot_lims=[9, 11.5, -10.5, -7.5], lower_limit=True, fig=fig, use_color_df=True) #, color_var='median_U_V'
        plot_a_vs_b_paper('median_log_mass', 'O3N2_metallicity', stellar_mass_label, metallicity_label, 'None', axis_obj=ax_metallicity, yerr=True, plot_lims=[9, 11.5, 8, 9], fig=fig)
        scale_aspect(ax_ssfr)
        scale_aspect(ax_metallicity)
        fig.savefig(imd.sed_paper_figures_dir + '/mass_ssfr_metallicity.pdf')
        plt.close('all')
    make_ssfr_mass_metallicity_fig()
    # Dust mass figure? Can we measure this?


    # UVJ and BPT 2-panel with all the stacks
    def make_uvj_bpt_fig():
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1],width_ratios=[1,1])
        ax_uvj = fig.add_subplot(gs[0, 0])
        ax_bpt = fig.add_subplot(gs[0, 1])
        # UVJs of all galaxies
        galaxy_uvj_df = ascii.read(imd.uvj_dir + '/galaxy_uvjs.csv').to_pandas()
        # UVJs of all composite SEDs
        composite_uvj_df = ascii.read(
        imd.composite_uvj_dir + '/composite_uvjs.csv').to_pandas()
        setup_uvj_plot(ax_uvj, galaxy_uvj_df, composite_uvj_df, include_unused_gals='No', paper_fig=True)
        ax_uvj.set_xlabel('V-J', fontsize=14)
        ax_uvj.set_ylabel('U-V', fontsize=14)

        scale_aspect(ax_uvj)
        scale_aspect(ax_bpt)
        fig.savefig(imd.sed_paper_figures_dir + '/uvj_bpt.pdf')
        plt.close('all')
    make_uvj_bpt_fig()

    pass


    
    
make_paper_plots(20, 'luminosity')