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
from plot_dust_model_vis import make_vis_plot
from compute_cluster_sfrs import compute_balmer_AV_from_ahalpha
from scipy.stats import linregress




def make_paper_plots(n_clusters, norm_method):
    # Overview figure
    # setup_figs(n_clusters, norm_method, bpt_color=True, paper_overview=True, prospector_spec=False)

    ### Potentially 4 panels? Or maybe different figures
    # Prospector AV vs Mass, and Balmer dec measured vs mass
    # AV vs Balmer decrement - how much extra attenuation?
    # Attenuation curve figure(s) - what controsl it
    make_AV_panel_fig()

    # Prospector Dust index fig
    # make_dust_index_fig()

    # SFR comparison between prospector and emission lines
    # make_SFR_compare_fig()

    #sfr/mass/uvj/bpt
    # make_sfr_mass_uvj_bpt_4panel(snr_thresh=3)

    # Dust model figure
    # make_dust_fig()


    # Dust mass figure? Can we measure this?
    pass




def make_AV_panel_fig():
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(2, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1,1],width_ratios=[1,1])
    ax_av_mass = fig.add_subplot(gs[0, 0])
    ax_balmer_mass = fig.add_subplot(gs[0, 1])
    ax_balmer_av_compare = fig.add_subplot(gs[1, 0])
    ax_av_difference = fig.add_subplot(gs[1, 1])
    prospector_dust2_label = 'Prospector Stellar A$_V$'
    av_difference_label = 'Nebular A$_V$ - Stellar A$_V$'
    plot_a_vs_b_paper('median_log_mass', 'Prospector_AV_50', stellar_mass_label, prospector_dust2_label, 'None', axis_obj=ax_av_mass, yerr=True, plot_lims=[9, 11.5, -0.2, 2.5], fig=fig, use_color_df=True) 
    plot_a_vs_b_paper('median_log_mass', 'balmer_av_with_limit', stellar_mass_label, balmer_av_label, 'None', axis_obj=ax_balmer_mass, yerr=True, plot_lims=[9, 11.5, -0.2, 5], fig=fig, use_color_df=True, lower_limit=True) 
    regress_res = find_best_fit('median_log_mass', 'balmer_av_with_limit', exclude_limit=True)
    x_regress = np.arange(9, 11.5, 0.1)
    ax_balmer_mass.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='gray', label=f'Linear Fit', ls='--')
    print(f'Best fit to nebular av vs mass: slope {regress_res.slope}, yint {regress_res.intercept}')
    # Shapley's data
    mosdef_data_mass = np.array([9.252764612954188, 9.73301737756714, 10.0173775671406, 10.437598736176936]) #Shapley 2022
    mosdef_data_decs = np.array([3.337349397590363, 3.4548192771084363, 3.7801204819277103, 4.512048192771086])
    mosdef_data_balmeravs = compute_balmer_av(mosdef_data_decs)
    ax_balmer_mass.plot(mosdef_data_mass, mosdef_data_balmeravs, color='black', marker='s', ms=10, mec='black', ls='--', zorder=1000000, label='z=2.3 MOSDEF (Shapley+ 2022)')
    ax_balmer_mass.legend(fontsize=14)
    plot_a_vs_b_paper('Prospector_AV_50', 'balmer_av_with_limit', prospector_dust2_label, balmer_av_label, 'None', axis_obj=ax_balmer_av_compare, yerr=True, plot_lims=[-0.2, 2, -0.2, 5], fig=fig, use_color_df=True, prospector_xerr=True, one_to_one=False, factor_of_2=True, lower_limit=True)
    regress_res = find_best_fit('Prospector_AV_50', 'balmer_av_with_limit', exclude_limit=True)
    x_regress = np.arange(-0.2, 2, 0.1)
    print(f'Best fit to Nebular vs Stellar av: slope {regress_res.slope}, yint {regress_res.intercept}')
    ax_balmer_av_compare.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='black', label='Linear fit', ls='--')
    ax_balmer_av_compare.legend(fontsize=14, loc=2)
    plot_a_vs_b_paper('computed_log_sfr_with_limit', 'AV_difference_with_limit', 'computed_log_sfr_with_limit', av_difference_label, 'None', axis_obj=ax_av_difference, yerr=True, fig=fig, use_color_df=True, lower_limit=True, plot_lims=[0, 2, -1, 3])
    regress_res = find_best_fit('computed_log_sfr_with_limit', 'AV_difference_with_limit', exclude_limit=True)
    x_regress = np.arange(-1, 3, 0.1)
    ax_av_difference.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='black', label='Linear fit', ls='--')
    ax_av_difference.legend(fontsize=14, loc=2)
    print(f'Best fit to Av difference vs SFR: slope {regress_res.slope}, yint {regress_res.intercept}')


    for ax in [ax_av_mass, ax_balmer_mass, ax_balmer_av_compare, ax_av_difference]:
        scale_aspect(ax)
        # ax.legend(fontsize=full_page_axisfont-4)
    fig.savefig(imd.sed_paper_figures_dir + '/dust_panel.pdf')
    plt.close('all')

def find_best_fit(x_col, y_col, exclude_limit=True):
    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    if exclude_limit == True:
        cluster_summary_df = cluster_summary_df[cluster_summary_df['flag_hb_limit']==0]
    xvals = cluster_summary_df[x_col]
    yvals = cluster_summary_df[y_col]
    regress_res = linregress(xvals, yvals)
    return regress_res



def make_SFR_compare_fig():
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(1, 1, left=0.11, right=0.96, bottom=0.12)
    ax_sfr = fig.add_subplot(gs[0, 0])
    
    plot_a_vs_b_paper('log_Prospector_ssfr50_multiplied_normalized', 'computed_log_sfr_with_limit', 'Prospector Normalized SED SFR', sfr_label, 'None', axis_obj=ax_sfr, yerr=True, lower_limit=True, plot_lims=[-1, 3, -1, 3], fig=fig, one_to_one=True, use_color_df=True, add_numbers=True)
    ax_sfr.tick_params(labelsize=full_page_axisfont)

    scale_aspect(ax_sfr)
    fig.savefig(imd.sed_paper_figures_dir + '/sfr_compare_normalized.pdf')

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
    
def make_sfr_mass_uvj_bpt_4panel(n_clusters=20, snr_thresh=2):
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(2, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1,1],width_ratios=[1,1])
    ax_ssfr = fig.add_subplot(gs[0, 0])
    ax_metallicity = fig.add_subplot(gs[0, 1])
    ax_uvj = fig.add_subplot(gs[1, 0])
    ax_bpt = fig.add_subplot(gs[1, 1])
    #SFR/Metallicity
    plot_a_vs_b_paper('median_log_mass', 'computed_log_ssfr_with_limit', stellar_mass_label, ssfr_label, 'None', axis_obj=ax_ssfr, yerr=True, plot_lims=[9, 11.5, -10.8, -7.5], lower_limit=True, fig=fig, use_color_df=True) #, color_var='median_U_V'
    plot_a_vs_b_paper('median_log_mass', 'O3N2_metallicity_upper_limit', stellar_mass_label, metallicity_label, 'None', axis_obj=ax_metallicity, yerr=True, plot_lims=[9, 11.5, 8.15, 9.17], fig=fig, upper_limit=True)
    for groupID in range(n_clusters):
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        group_df['hb_SNR'] = group_df['hb_flux'] / group_df['err_hb_flux']
        group_df['nii_6585_SNR'] = group_df['nii_6585_flux'] / group_df['err_nii_6585_flux']
        hb_detected_rows = group_df['hb_SNR']>snr_thresh
        nii_detected_rows = group_df['nii_6585_SNR']>snr_thresh
        both_detected = np.logical_and(nii_detected_rows, hb_detected_rows)
        group_df['log_recomputed_ssfr'] = np.log10(group_df['recomputed_sfr']/(10**group_df['log_mass']))
        ax_ssfr.plot(group_df['log_mass'], group_df['log_recomputed_ssfr'], color=grey_point_color, markersize=grey_point_size, marker='o', ls='None')
        #Compute metallicity
        group_df['N2Ha'] = group_df['nii_6585_flux'] / group_df['ha_flux']
        group_df['O3Hb'] = group_df['oiii_5008_flux'] / group_df['hb_flux']
        group_df['O3N2_metallicity'] = 8.97-0.39*np.log10(group_df['O3Hb'] / group_df['N2Ha']) 
        
        ax_metallicity.plot(group_df[both_detected]['log_mass'], group_df[both_detected]['O3N2_metallicity'], color=grey_point_color, markersize=grey_point_size, marker='o', ls='None')
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
    plot_bpt(axis_obj=ax_bpt, skip_gals=True, add_background=True, snr_background=snr_thresh)
    add_composite_bpts(ax_bpt)
    ax_bpt.set_xlabel('log(N[II] 6583 / H$\\alpha$)', fontsize=full_page_axisfont)
    ax_bpt.set_ylabel('log(O[III] 5007 / H$\\beta$)', fontsize=full_page_axisfont)

    ax_uvj.tick_params(labelsize=full_page_axisfont)
    ax_bpt.tick_params(labelsize=full_page_axisfont)

    scale_aspect(ax_uvj)
    scale_aspect(ax_bpt)
    fig.savefig(imd.sed_paper_figures_dir + '/mass_sfr_uvj_bpt.pdf')

def make_dust_fig():
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(1, 1, left=0.11, right=0.96, bottom=0.12)
    ax_dust_model = fig.add_subplot(gs[0, 0])
    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    logsfrs = cluster_summary_df['computed_log_sfr_with_limit']
    sfrs = 10**logsfrs
    metallicities = cluster_summary_df['O3N2_metallicity']
    res = cluster_summary_df['median_re']
    balmer_av = cluster_summary_df['balmer_av_with_limit']

    a = 2.15
    n = 1.4
    x_axis_vals = 10**(a*metallicities) * sfrs**(1/n)
    make_vis_plot(x_axis_vals, balmer_av, '', '', '', axis_obj=ax_dust_model, fig=fig)
    
    ax_dust_model.set_ylim(-0.1,4)
    ax_dust_model.tick_params(labelsize=full_page_axisfont)
    ax_dust_model.set_xlabel('$10^{a\\times metallicity}\\times SFR^{1/n}$', fontsize=full_page_axisfont)
    ax_dust_model.set_ylabel('Balmer AV', fontsize=full_page_axisfont)

    from plot_sfr_metallicity import sanders_plane
    from dust_model import const2
    def compute_sanders_metals(log_mass, fm_s):
        '''
        Parameters:
        log_mass: Log stellar mass
        fm_s: log sfrs (array)
        '''
        fm_m = (log_mass-10)*np.ones(len(fm_s))
        fm_metals = sanders_plane(log_mass, fm_s)
        return fm_metals
    sanders_log_sfrs = np.arange(0.5, 1.7, 0.1)
    # masses = [9, 9.5, 10, 10.5, 11]
    # colors = ['red', 'blue', 'green', 'black', 'orange']
    # sfregions = [1, 1, 1, 1, 1]
    masses = [9.5, 10, 10.5]
    colors = ['black', 'red', 'blue']
    sfregions = [1, 1, 1]
    for i in range(len(masses)):
        sanders_log_mass = masses[i]
     
        sanders_metallicities = compute_sanders_metals(sanders_log_mass, sanders_log_sfrs)
        
        sanders_x_axis_vals = 10**(a*sanders_metallicities) * ((10**sanders_log_sfrs)/(sfregions[i]**2))**(1/n)
        sanders_yvals = sanders_x_axis_vals*const2
        ax_dust_model.plot(sanders_x_axis_vals, sanders_yvals, ls='--', color=colors[i])


    ax_dust_model.set_xscale('log')

    fig.savefig(imd.sed_paper_figures_dir + '/dust_model.pdf')

def make_dust_index_fig():
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(1, 1, left=0.11, right=0.96, bottom=0.12)
    ax_dust_index = fig.add_subplot(gs[0, 0])
    
    dust_index_label = 'Prospector dust_index'

    plot_a_vs_b_paper('median_log_mass', 'dustindex50', stellar_mass_label, dust_index_label, 'None', axis_obj=ax_dust_index, yerr=True, plot_lims=[9, 11.5, -1.7, 1.0], fig=fig, use_color_df=True, lower_limit=False)
    ax_dust_index.tick_params(labelsize=full_page_axisfont)

    scale_aspect(ax_dust_index)
    fig.savefig(imd.sed_paper_figures_dir + '/dust_index.pdf')

make_paper_plots(20, 'luminosity')