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
from balmer_avs import compute_balmer_av




def make_paper_plots(n_clusters, norm_method):
    # Overview figure
    setup_figs(n_clusters, norm_method, bpt_color=True, paper_overview=True, prospector_spec=False)

    ### Potentially 4 panels? Or maybe different figures
    # Prospector AV vs Mass, and Balmer dec measured vs mass
    # AV vs Balmer decrement - how much extra attenuation?
    # Attenuation curve figure(s) - what controsl it
    make_AV_panel_fig()

    # SFR comparison between prospector and emission lines
    make_SFR_compare_fig()

    #sfr/mass/uvj/bpt
    make_sfr_mass_uvj_bpt_4panel()


    # Dust mass figure? Can we measure this?
    pass




def make_AV_panel_fig():
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(2, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1,1],width_ratios=[1,1])
    ax_av_mass = fig.add_subplot(gs[0, 0])
    ax_balmer_mass = fig.add_subplot(gs[0, 1])
    ax_balmer_av_compare = fig.add_subplot(gs[1, 0])
    ax_dust_index = fig.add_subplot(gs[1, 1])
    prospector_dust2_label = 'Prospector dust2'
    dust_index_label = 'Prospector dust_index'
    plot_a_vs_b_paper('median_log_mass', 'dust2_50', stellar_mass_label, prospector_dust2_label, 'None', axis_obj=ax_av_mass, yerr=True, plot_lims=[9, 11.5, -0.2, 2.5], fig=fig, use_color_df=True) 
    plot_a_vs_b_paper('median_log_mass', 'balmer_av_with_limit', stellar_mass_label, balmer_av_label, 'None', axis_obj=ax_balmer_mass, yerr=True, plot_lims=[9, 11.5, -0.2, 5], fig=fig, use_color_df=True, lower_limit=True) 
    # Shapley's data
    mosdef_data_mass = np.array([9.252764612954188, 9.73301737756714, 10.0173775671406, 10.437598736176936]) #Shapley 2022
    mosdef_data_decs = np.array([3.337349397590363, 3.4548192771084363, 3.7801204819277103, 4.512048192771086])
    mosdef_data_balmeravs = compute_balmer_av(mosdef_data_decs)
    ax_balmer_mass.plot(mosdef_data_mass, mosdef_data_balmeravs, color='black', marker='s', ms=10, mec='black', ls='None', zorder=1000000, label='z=2.3 MOSDEF (Shapley+ 2022)')
    ax_balmer_mass.legend(fontsize=14)
    plot_a_vs_b_paper('dust2_50', 'balmer_av_with_limit', prospector_dust2_label, balmer_av_label, 'None', axis_obj=ax_balmer_av_compare, yerr=True, plot_lims=[-0.2, 2, -0.2, 5], fig=fig, use_color_df=True, prospector_xerr=True, one_to_one=True, factor_of_2=True, lower_limit=True)
    plot_a_vs_b_paper('median_log_mass', 'dustindex50', stellar_mass_label, dust_index_label, 'None', axis_obj=ax_dust_index, yerr=True, plot_lims=[9, 11.5, -1.5, 0.5], fig=fig, use_color_df=True) 
    for ax in [ax_av_mass, ax_balmer_mass, ax_balmer_av_compare, ax_dust_index]:
        scale_aspect(ax)
        # ax.legend(fontsize=full_page_axisfont-4)
    fig.savefig(imd.sed_paper_figures_dir + '/dust_panel.pdf')
    plt.close('all')


def make_SFR_compare_fig():
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(1, 1, left=0.11, right=0.96, bottom=0.12)
    ax_sfr = fig.add_subplot(gs[0, 0])
    plot_a_vs_b_paper('prospector_log_ssfr', 'computed_log_ssfr_with_limit', ssfr_label, ssfr_label, 'None', axis_obj=ax_sfr, yerr=True, lower_limit=True, plot_lims=[-11, -7, -11, -7], fig=fig, one_to_one=True, use_color_df=True)


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
    
def make_sfr_mass_uvj_bpt_4panel(n_clusters=20):
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(2, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1,1],width_ratios=[1,1])
    ax_ssfr = fig.add_subplot(gs[0, 0])
    ax_metallicity = fig.add_subplot(gs[0, 1])
    ax_uvj = fig.add_subplot(gs[1, 0])
    ax_bpt = fig.add_subplot(gs[1, 1])
    #SFR/Metallicity
    plot_a_vs_b_paper('median_log_mass', 'computed_log_ssfr_with_limit', stellar_mass_label, ssfr_label, 'None', axis_obj=ax_ssfr, yerr=True, plot_lims=[9, 11.5, -10.8, -7.5], lower_limit=True, fig=fig, use_color_df=True) #, color_var='median_U_V'
    plot_a_vs_b_paper('median_log_mass', 'O3N2_metallicity', stellar_mass_label, metallicity_label, 'None', axis_obj=ax_metallicity, yerr=True, plot_lims=[9, 11.5, 8, 9], fig=fig)
    for groupID in range(n_clusters):
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        group_df['log_recomputed_ssfr'] = np.log10(group_df['recomputed_sfr']/(10**group_df['log_mass']))
        ax_ssfr.plot(group_df['log_mass'], group_df['log_recomputed_ssfr'], color=grey_point_color, markersize=grey_point_size, marker='o', ls='None')
        # ok_balmer_rows = np.logical_and(group_df['ha_detflag_sfr']==0, group_df['hb_detflag_sfr']==0)
        # ax_ssfr.plot(group_df[ok_balmer_rows]['log_mass'], group_df[ok_balmer_rows]['log_recomputed_ssfr'], color='black', markersize=grey_point_size, marker='o', ls='None')

    add_sanders_metallicity(ax_metallicity)
    # add_leja_sfms(ax_ssfr, mode='ridge')
    add_leja_sfms(ax_ssfr, mode='mean')
    for ax in [ax_ssfr, ax_metallicity]:
        scale_aspect(ax)
        ax.legend(fontsize=full_page_axisfont-4)

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
    fig.savefig(imd.sed_paper_figures_dir + '/mass_sfr_uvj_bpt.pdf')

make_paper_plots(20, 'luminosity')