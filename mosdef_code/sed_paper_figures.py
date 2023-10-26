import initialize_mosdef_dirs as imd
from overview_plot_of_clusters import make_overview_plot_clusters, setup_figs
from plot_cluster_a_vs_b import plot_a_vs_b_paper
import matplotlib.pyplot as plt
from plot_vals import *
from matplotlib.gridspec import GridSpec
from uvj_clusters import setup_uvj_plot
from bpt_clusters_singledf import plot_bpt, add_composite_bpts
from astropy.io import ascii
from plot_cluster_a_vs_b import add_leja_sfms
from compute_metals_ssfr import add_sanders_metallicity




def make_paper_plots(n_clusters, norm_method):
    # Overview figure
    # setup_figs(n_clusters, norm_method, bpt_color=True, paper_overview=False, prospector_spec=False)

    ### Potentially 4 panels? Or maybe different figures
    # Prospector AV vs Mass, and Balmer dec measured vs mass
    # AV vs Balmer decrement - how much extra attenuation?
    # Attenuation curve figure(s) - what controsl it
    make_AV_panel_fig()

    # SFR comparison between prospector and emission lines
    pass


    #ssfr vs mass, metallicity vs mass
    # make_ssfr_mass_metallicity_fig()

    # Dust mass figure? Can we measure this?
    pass

    # UVJ and BPT 2-panel with all the stacks
    # make_uvj_bpt_fig()

    pass

def make_AV_panel_fig():
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1,1],width_ratios=[1,1])
    ax_av_mass = fig.add_subplot(gs[0, 0])
    ax_balmer_mass = fig.add_subplot(gs[0, 1])

    plot_a_vs_b_paper('median_log_mass', 'balmer_av', stellar_mass_label, balmer_av_label, 'None', axis_obj=ax_balmer_mass, yerr=True, plot_lims=[9, 11.5, 2, 12], fig=fig, use_color_df=True) #, color_var='median_U_V'
    for ax in [ax_av_mass, ax_balmer_mass]:
        scale_aspect(ax)
        # ax.legend(fontsize=full_page_axisfont-4)
    fig.savefig(imd.sed_paper_figures_dir + '/dust_panel.pdf')
    plt.close('all')


def make_ssfr_mass_metallicity_fig():
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1],width_ratios=[1,1])
    ax_ssfr = fig.add_subplot(gs[0, 0])
    ax_metallicity = fig.add_subplot(gs[0, 1])
    plot_a_vs_b_paper('median_log_mass', 'computed_log_ssfr_with_limit', stellar_mass_label, ssfr_label, 'None', axis_obj=ax_ssfr, yerr=True, plot_lims=[9, 11.5, -10.5, -7.5], lower_limit=True, fig=fig, use_color_df=True) #, color_var='median_U_V'
    plot_a_vs_b_paper('median_log_mass', 'O3N2_metallicity', stellar_mass_label, metallicity_label, 'None', axis_obj=ax_metallicity, yerr=True, plot_lims=[9, 11.5, 8, 9], fig=fig)
    gal_df = ascii.read(imd.loc_filtered_gal_df).to_pandas()
    gal_df['log_use_ssfr'] = np.log10(gal_df['use_sfr']/(10**gal_df['log_mass']))
    ax_ssfr.plot(gal_df['log_mass'], gal_df['log_use_ssfr'], color=grey_point_color, markersize=grey_point_size, marker='o', ls='None')
    add_sanders_metallicity(ax_metallicity)
    add_leja_sfms(ax_ssfr)
    for ax in [ax_ssfr, ax_metallicity]:
        scale_aspect(ax)
        ax.legend(fontsize=full_page_axisfont-4)
    fig.savefig(imd.sed_paper_figures_dir + '/mass_ssfr_metallicity.pdf')
    plt.close('all')

def make_uvj_bpt_fig():
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1],width_ratios=[1,1])
    ax_uvj = fig.add_subplot(gs[0, 0])
    ax_bpt = fig.add_subplot(gs[0, 1])
    # UVJs of all galaxies
    galaxy_uvj_df = ascii.read(imd.uvj_dir + '/galaxy_uvjs.csv').to_pandas()
    # UVJs of all composite SEDs
    composite_uvj_df = ascii.read(imd.composite_uvj_dir + '/composite_uvjs.csv').to_pandas()
    setup_uvj_plot(ax_uvj, galaxy_uvj_df, composite_uvj_df, include_unused_gals='No', paper_fig=True)
    ax_uvj.set_xlabel('V-J', fontsize=full_page_axisfont)
    ax_uvj.set_ylabel('U-V', fontsize=full_page_axisfont)

    #BPT Diagram
    plot_bpt(axis_obj=ax_bpt, skip_gals=True, add_background=True)
    add_composite_bpts(ax_bpt)
    ax_bpt.set_xlabel('log(N[II] 6583 / H$\\alpha$)', fontsize=full_page_axisfont)
    ax_bpt.set_ylabel('log(O[III] 5007 / H$\\beta$)', fontsize=full_page_axisfont)

    ax_uvj.tick_params(labelsize=full_page_axisfont)
    ax_bpt.tick_params(labelsize=full_page_axisfont)

    scale_aspect(ax_uvj)
    scale_aspect(ax_bpt)
    fig.savefig(imd.sed_paper_figures_dir + '/uvj_bpt.pdf')
    plt.close('all')
    


make_paper_plots(20, 'luminosity')