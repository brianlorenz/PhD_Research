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
from scipy.optimize import curve_fit
from dust_model import sanders_plane
from compute_cluster_sfrs import draw_asymettric_error
from prospector_plot import load_obj
from generate_paper_table_composites import generate_sed_paper_table
import pandas as pd

def compute_metals(log_mass, fm_s):
    '''
    Parameters:
    log_mass: Log stellar mass
    fm_s: log sfrs (array)
    '''
    fm_metals = sanders_plane(log_mass, fm_s)
    return fm_metals




def make_paper_plots(n_clusters, norm_method):
    # Overview figure
    # generate_sed_paper_table()
    # setup_figs(n_clusters, norm_method, bpt_color=True, paper_overview=True, prospector_spec=False)
    # make_av_comparison()
    # make_SFR_compare_fig()
    # make_prospector_overview_fig('fixedmet_agn_set')
    # make_sfr_mass_uvj_bpt_4panel(snr_thresh=3)
    # make_mass_metal_sfr_fig()
    


    # # # Dust model figure
    # make_dust_fig()

    ### Potentially 4 panels? Or maybe different figures
    # Prospector AV vs Mass, and Balmer dec measured vs mass
    # AV vs Balmer decrement - how much extra attenuation?
    # Attenuation curve figure(s) - what controsl it
    # make_AV_panel_fig()

    # Prospector Dust index fig
    # make_dust_index_fig()

    # # SFR comparison between prospector and emission lines
    # make_prospector_compare_met_fig()

    
    # # Metals/SFR/both dust figure

    # # hb percentage figure
    make_hb_percentage_fig()

    # Dust mass figure? Can we measure this?
    pass

prospector_dust2_label = 'Prospector Stellar A$_V$'

def line(x, a, b):
    return a * x + b

def make_prospector_overview_fig(run_name):
    fontsize = 20
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(4, 5, left=0.02, right=0.98, wspace=0.21, hspace=0.04, top=0.98, bottom=0.02)
    cluster_summary_df = imd.read_cluster_summary_df()
    paperID = 1
    for row in range(4):
        for col in range(5):
            ax = fig.add_subplot(gs[row, col])
            groupID = cluster_summary_df[cluster_summary_df['paperID'] == paperID]['groupID'].iloc[0]
            save_str = f'group{groupID}'
            obs = load_obj(f'{save_str}_obs', run_name)
            color = get_row_color(groupID)


            spec_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs' + 
                                f'/{save_str}_spec.csv').to_pandas()
            phot_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs' +
                                f'/{save_str}_phot.csv').to_pandas()
            lines_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs' +
                                f'/{save_str}_lines.csv').to_pandas()
            
            phot_5000_idx = np.logical_and(phot_df['rest_wavelength']>4500, phot_df['rest_wavelength']<5500)
            phot_5000 = phot_df[phot_5000_idx].iloc[0]['phot50_flambda']
            # breakpoint()
            scale = 1/(phot_5000*5000)
            for i in range(len(obs['err_f_lambda'])):
                if i == len(obs['err_f_lambda']) - 1:
                    continue
                if obs['err_f_lambda'][i+1] > 10 * obs['err_f_lambda'][i]:
                    obs['err_f_lambda'][i+1] = obs['err_f_lambda'][i]
            phot_idxs = phot_df['rest_wavelength'] > 1500

            start_spec = phot_df['rest_wavelength'].iloc[0]
            end_spec = phot_df['rest_wavelength'].iloc[-1]
            spec_idxs = np.logical_and(spec_df['rest_wavelength'] > start_spec, spec_df['rest_wavelength'] < end_spec)
            
            rest_frame_original_phot = obs['f_lambda']*(1+obs['z'])
            rest_frame_original_phot_errs = obs['err_f_lambda']*(1+obs['z'])
            errs_low_flambda = phot_df['rest_wavelength']*rest_frame_original_phot - phot_df['rest_wavelength']*rest_frame_original_phot_errs
            errs_high_flambda = phot_df['rest_wavelength']*rest_frame_original_phot + phot_df['rest_wavelength']*rest_frame_original_phot_errs
            # Points with errorbars for obs
            ax.errorbar(phot_df['rest_wavelength'], scale * phot_df['rest_wavelength'] * rest_frame_original_phot, color='black', yerr=phot_df['rest_wavelength'] * rest_frame_original_phot_errs * scale, ls='None', marker='o', label='Composite SED', zorder=1, mec='black', mfc=color, markersize=10)
            # Just points and shaded region for observations
            # ax.plot(phot_df['rest_wavelength'], phot_df['rest_wavelength'] * rest_frame_original_phot, color='black', ls='None', marker='o', label='Observations', zorder=1)            
            # errs_low_flambda = errs_low_flambda.to_list()
            # errs_high_flambda = errs_high_flambda.to_list()
            # ax.fill_between(phot_df['rest_wavelength'], errs_low_flambda, errs_high_flambda, facecolor="gray", alpha=0.7)


            y_model = np.array(phot_df['rest_wavelength'][phot_idxs]
                        * phot_df['phot50_flambda'][phot_idxs])
            y_model_16 = phot_df['rest_wavelength'][phot_idxs] * phot_df['phot16_flambda'][phot_idxs]
            y_model_84 = phot_df['rest_wavelength'][phot_idxs] * phot_df['phot84_flambda'][phot_idxs]
            model_errs = np.vstack((y_model - y_model_16, y_model_84 - y_model))
            # Model points with errorbar, or can set points to none and use just a line
            # ax.errorbar(np.array(phot_df['rest_wavelength']), y_model,
            #             ls='None', lw=3, marker='o', yerr=model_errs, color='blue', label='Model')
            
            #Model fill between
            ax.fill_between(phot_df['rest_wavelength'][phot_idxs], scale*y_model_16, scale*y_model_84, alpha=0.6, color='grey', label='Model')
            
            ax.axvspan(phot_df['rest_wavelength'].iloc[0]-100, 1500, alpha=0.3, color='red')
            if col == 0:
                ax.set_ylabel("$\lambda$ F$_\lambda$ (norm)", fontsize = fontsize)
            if row == 3:
                ax.set_xlabel("Wavelength ($\AA$)", fontsize=fontsize)
            if row != 3:
                ax.tick_params(labelbottom=False)    

            ax.tick_params(labelsize=fontsize)

            xtext = 0.638
            if paperID < 10: 
                xtext = xtext + 0.04
            ax.text(xtext, 0.927, f'Group {paperID}', transform=ax.transAxes, fontsize=fontsize)

            ax.set_ylim(0.8 * np.percentile(scale*phot_df['rest_wavelength'] * rest_frame_original_phot, 1),
                     1.25 * np.percentile(scale*phot_df['rest_wavelength'] * rest_frame_original_phot, 99))

            
            ax.set_xlim(phot_df['rest_wavelength'].iloc[0]-30, phot_df['rest_wavelength'].iloc[-1] + 3000)
            ax.set_xscale('log')
            # scale_aspect(ax)
            if paperID == 1:
                ax.legend(fontsize=16, loc=(0.28, 0.69))

            paperID = paperID + 1
    fig.savefig(imd.sed_paper_figures_dir + '/prospector_overview.pdf', bbox_inches='tight')



def make_prospector_compare_met_fig(n_groups=20):
    fig = plt.figure(figsize=(6.2, 6))
    gs = GridSpec(1, 1, left=0.11, right=0.96, bottom=0.12)
    ax_compare = fig.add_subplot(gs[0, 0])

    cluster_summary_df = imd.read_cluster_summary_df()
    
    prospector_met = cluster_summary_df['logzsol50'] + 8.69
    ax_compare.plot(cluster_summary_df['O3N2_metallicity'], prospector_met, marker='o', color='black', ls='None')
    ax_compare.plot([-10, 10], [-10, 10], marker='None', color='red', ls='--')
    
    ax_compare.set_xlabel('Spectrum O3N2 metallicity', fontsize=full_page_axisfont)
    ax_compare.set_ylabel('Prospector SED metallicity', fontsize=full_page_axisfont)
    ax_compare.tick_params(labelsize=full_page_axisfont)
    ax_compare.set_xlim(8.2, 8.9)
    ax_compare.set_ylim(7.1, 9)

    scale_aspect(ax_compare)
    fig.savefig(imd.sed_paper_figures_dir + '/prospector_met_compare.pdf', bbox_inches='tight')



def make_hb_percentage_fig(n_groups=20):
    fig = plt.figure(figsize=(6.2, 6))
    gs = GridSpec(1, 1, left=0.11, right=0.96, bottom=0.12)
    ax_hb = fig.add_subplot(gs[0, 0])

    cluster_summary_df = imd.read_cluster_summary_df()
    hb_fracs = []
    ha_good = []
    ha_good_hb_bad = []
    missing_hbs = []
    n_tot = []
    median_indiv_a_balmers = []
    for groupID in range(n_groups):
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        n_hb_good = len(group_df[group_df['hb_detflag_sfr'] == 0])
        ha_good_df = group_df[group_df['ha_detflag_sfr'] == 0]
        n_ha_good = len(ha_good_df)
        ha_good.append(n_ha_good)
        n_missing_hb = len(group_df[group_df['hb_detflag_sfr'] != 0])
        n_ha_good_hb_bad = len(ha_good_df[ha_good_df['hb_detflag_sfr'] != 0])
        ha_good_hb_bad.append(n_ha_good_hb_bad)
        missing_hbs.append(n_missing_hb)
        n_tot.append(len(group_df))
        frac_hb = n_hb_good / len(group_df)
        hb_fracs.append(frac_hb)
        def compute_balmer_av(balmer_dec):
            balmer_av = 4.05*1.97*np.log10(balmer_dec/2.86)
            return balmer_av
        hb_goods = group_df[group_df['hb_detflag_sfr'] == 0]
        ha_goods = hb_goods[hb_goods['ha_detflag_sfr'] == 0]
        abalmers = compute_balmer_av(ha_goods['balmer_dec'])
        
        median_abalmer = np.median(abalmers)
        print(f'{groupID}        {median_abalmer}')
        median_indiv_a_balmers.append(median_abalmer)
        
    breakpoint()
    a_balmer_difference = cluster_summary_df['balmer_av_with_limit'] - median_indiv_a_balmers
    hbsnr = cluster_summary_df['hb_snr']
    # ax_hb.plot(hb_fracs, hbsnr, marker='o', color='black', ls='None')
    ax_hb.plot(hbsnr, a_balmer_difference, marker='o', color='black', ls='None')
    # plot_a_vs_b_paper('log_Prospector_ssfr50_multiplied_normalized', 'computed_log_sfr_with_limit', 'Prospector Normalized SED SFR', 'log$_{10}$(H$\\mathrm{\\alpha}$ SFR) (M$_\odot$ / yr)', 'None', axis_obj=ax_sfr, yerr=True, lower_limit=180, plot_lims=[-1, 2.5, 0.1, 2.1], fig=fig, one_to_one=True, use_color_df=True, add_numbers=False)
    # ax_hb.legend(fontsize=16, loc=2)
    ax_hb.set_ylabel('Abalmer diff', fontsize=full_page_axisfont)
    ax_hb.set_xlabel('Stack Hbeta SNR', fontsize=full_page_axisfont)
    ax_hb.tick_params(labelsize=full_page_axisfont)

    scale_aspect(ax_hb)
    fig.savefig(imd.sed_paper_figures_dir + '/hb_frac_fig_snr.pdf', bbox_inches='tight')



def make_mass_metal_sfr_fig():
    fig = plt.figure(figsize=(18, 12))
    rfont = 18
    gs = GridSpec(2, 3, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1,1],width_ratios=[1,1,1])    
    ax_stellarav_mass = fig.add_subplot(gs[0, 0])
    ax_balmer_mass = fig.add_subplot(gs[1, 0])
    ax_balmer_sfr = fig.add_subplot(gs[1, 1])
    ax_balmer_metal = fig.add_subplot(gs[1, 2])
    ax_stellarav_sfr = fig.add_subplot(gs[0, 1])
    ax_stellarav_metal = fig.add_subplot(gs[0, 2])
    # ax_metal_sfr = fig.add_subplot(gs[0, 2])
    plot_a_vs_b_paper('computed_log_sfr_with_limit', 'Prospector_AV_50', sfr_label, prospector_dust2_label, 'None', axis_obj=ax_stellarav_sfr, yerr=True, plot_lims=[0, 2, -0.2, 2.5], fig=fig, lower_limit=90, use_color_df=True) 
    plot_a_vs_b_paper('O3N2_metallicity', 'Prospector_AV_50', metallicity_label, prospector_dust2_label, 'None', axis_obj=ax_stellarav_metal, yerr=True, plot_lims=[8, 9, -0.2, 2.5], fig=fig, lower_limit=270, use_color_df=True)
    plot_a_vs_b_paper('computed_log_sfr_with_limit', 'balmer_av_with_limit', sfr_label, balmer_av_label, 'None', axis_obj=ax_balmer_sfr, yerr=True, plot_lims=[0, 2, -0.2, 5], fig=fig, lower_limit=135, use_color_df=True) 
    plot_a_vs_b_paper('O3N2_metallicity', 'balmer_av_with_limit', metallicity_label, balmer_av_label, 'None', axis_obj=ax_balmer_metal, yerr=True, plot_lims=[8, 9, -0.2, 5], fig=fig, lower_limit=225, use_color_df=True)
    plot_a_vs_b_paper('median_log_mass', 'Prospector_AV_50', stellar_mass_label, prospector_dust2_label, 'None', axis_obj=ax_stellarav_mass, yerr=True, plot_lims=[9, 11.5, -0.2, 2.5], fig=fig, use_color_df=True, lower_limit=3) 
    plot_a_vs_b_paper('median_log_mass', 'balmer_av_with_limit', stellar_mass_label, balmer_av_label, 'None', axis_obj=ax_balmer_mass, yerr=True, plot_lims=[9, 11.5, -0.2, 5], fig=fig, use_color_df=True, lower_limit=180) 
    # regress_res = find_best_fit('median_log_mass', 'balmer_av_with_limit', exclude_limit=True)
    x_regress = np.arange(9, 11.8, 0.1)
    regress_res, points_16, points_84 = bootstrap_fit('median_log_mass', 'balmer_av_with_limit', x_regress, exclude_limit=True)
    # ax_balmer_mass.plot(x_regress, yints[0] + slopes[0]*x_regress, color='black', ls='--')
    # ax_balmer_mass.plot(x_regress, yints[1] + slopes[1]*x_regress, color='green', ls='-')
    # ax_balmer_mass.plot(x_regress, yints[2] + slopes[2]*x_regress, color='green', ls='-')
    ax_balmer_mass.fill_between(x_regress, points_16, points_84, facecolor="gray", alpha=0.3)
    ax_balmer_mass.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='black', ls='--')
    def add_r(ax, regress):
        ax.text(0.03, 0.93, f"r={round(regress.rvalue, 2):.2f}", fontsize=rfont, transform = ax.transAxes)
    add_r(ax_balmer_mass, regress_res)
    print(f'Best fit to nebular av vs mass: slope {regress_res.slope}, yint {regress_res.intercept}')

    regress_res = find_best_fit('computed_log_sfr_with_limit', 'balmer_av_with_limit', exclude_limit=True)
    x_regress = np.arange(-0.1, 3, 0.1)
    regress_res, points_16, points_84 = bootstrap_fit('computed_log_sfr_with_limit', 'balmer_av_with_limit', x_regress, exclude_limit=True)
    add_r(ax_balmer_sfr, regress_res)
    # ax_av_sfr.plot(x_regress, yints[1] + slopes[1]*x_regress, color='green', ls='-')
    # ax_av_sfr.plot(x_regress, yints[2] + slopes[2]*x_regress, color='green', ls='-')
    ax_balmer_sfr.fill_between(x_regress, points_16, points_84, facecolor="gray", alpha=0.3)
    ax_balmer_sfr.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='black', ls='--')
    
    print(f'Best fit to nebular av vs sfr: slope {regress_res.slope}, yint {regress_res.intercept}')
    # Shapley's data
    mosdef_data_mass = np.array([9.252764612954188, 9.73301737756714, 10.0173775671406, 10.437598736176936]) #Shapley 2022
    mosdef_data_decs = np.array([3.337349397590363, 3.4548192771084363, 3.7801204819277103, 4.512048192771086])
    mosdef_data_balmeravs = compute_balmer_av(mosdef_data_decs)
    ax_balmer_mass.plot(mosdef_data_mass, mosdef_data_balmeravs, color='black', marker='s', ms=10, mec='black', ls='--', zorder=1000000, label='z=2.3 MOSDEF (Shapley+ 2022)')
    ax_balmer_mass.legend(loc = (0.03, 0.82), fontsize=14)
    # plot_a_vs_b_paper('computed_log_sfr_with_limit', 'O3N2_metallicity', metallicity_label, sfr_label, 'None', axis_obj=ax_metal_sfr, yerr=True, plot_lims=[0, 2, 8, 9], fig=fig, color_var='balmer_av', lower_limit=225, use_color_df=False)

    # Fit r values for the rest
    x_regress = np.arange(9, 11.8, 0.1) # Mass axis
    regress_res, points_16, points_84 = bootstrap_fit('median_log_mass', 'Prospector_AV_50', x_regress, exclude_limit=True, bootstrap=10)
    add_r(ax_stellarav_mass, regress_res)
    x_regress = np.arange(-0.1, 3, 0.1)
    regress_res, points_16, points_84 = bootstrap_fit('computed_log_sfr_with_limit', 'Prospector_AV_50', x_regress, exclude_limit=True, bootstrap=10)
    add_r(ax_stellarav_sfr, regress_res)
    x_regress = np.arange(7.9, 9.2, 0.1)
    regress_res, points_16, points_84 = bootstrap_fit('O3N2_metallicity', 'Prospector_AV_50', x_regress, exclude_limit=True, bootstrap=10)
    add_r(ax_stellarav_metal, regress_res)
    regress_res, points_16, points_84 = bootstrap_fit('O3N2_metallicity', 'balmer_av_with_limit', x_regress, exclude_limit=True, bootstrap=10)
    add_r(ax_balmer_metal, regress_res)

    # Add FMR
    fm_s = np.arange(-0.1, 4)
    log_mass_low = 9.5
    log_mass_high = 10.5
    log_masses = [log_mass_low, log_mass_high]
    # def add_sanders_fmr(log_mass):
    #     metals = compute_metals(log_mass, fm_s)
    #     ax_metal_sfr.plot(fm_s, metals, ls='--', color='black', marker='None')
    # for mass in log_masses:  
    #     add_sanders_fmr(mass)



    for ax in [ax_balmer_sfr, ax_balmer_metal, ax_stellarav_metal, ax_stellarav_sfr, ax_stellarav_mass, ax_balmer_mass]:
        scale_aspect(ax)
        # ax.legend(fontsize=full_page_axisfont-4)
    fig.savefig(imd.sed_paper_figures_dir + '/dust_mass_sfr_met.pdf', bbox_inches='tight')
    plt.close('all')


# def make_metal_sfr_fig():
#     fig = plt.figure(figsize=(12, 12))
#     gs = GridSpec(2, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1,1],width_ratios=[1,1])    
#     ax_av_sfr = fig.add_subplot(gs[0, 0])
#     ax_av_metal = fig.add_subplot(gs[0, 1])
#     ax_stellarav_sfr = fig.add_subplot(gs[1, 0])
#     ax_stellarav_metal = fig.add_subplot(gs[1, 1])
#     # ax_metal_sfr = fig.add_subplot(gs[0, 2])
#     plot_a_vs_b_paper('computed_log_sfr_with_limit', 'Prospector_AV_50', sfr_label, prospector_dust2_label, 'None', axis_obj=ax_av_sfr, yerr=True, plot_lims=[0, 2, -0.2, 2], fig=fig, lower_limit=270, use_color_df=True) 
#     plot_a_vs_b_paper('O3N2_metallicity', 'Prospector_AV_50', metallicity_label, prospector_dust2_label, 'None', axis_obj=ax_av_metal, yerr=True, plot_lims=[8, 9, -0.2, 2], fig=fig, lower_limit=90, use_color_df=True)
#     plot_a_vs_b_paper('computed_log_sfr_with_limit', 'balmer_av_with_limit', sfr_label, balmer_av_label, 'None', axis_obj=ax_stellarav_sfr, yerr=True, plot_lims=[0, 2, -0.2, 5], fig=fig, lower_limit=315, use_color_df=True) 
#     plot_a_vs_b_paper('O3N2_metallicity', 'balmer_av_with_limit', metallicity_label, balmer_av_label, 'None', axis_obj=ax_stellarav_metal, yerr=True, plot_lims=[8, 9, -0.2, 5], fig=fig, lower_limit=45, use_color_df=True)

#     # plot_a_vs_b_paper('computed_log_sfr_with_limit', 'O3N2_metallicity', metallicity_label, sfr_label, 'None', axis_obj=ax_metal_sfr, yerr=True, plot_lims=[0, 2, 8, 9], fig=fig, color_var='balmer_av', lower_limit=225, use_color_df=False)

#     # Add FMR
#     fm_s = np.arange(-0.1, 4)
#     log_mass_low = 9.5
#     log_mass_high = 10.5
#     log_masses = [log_mass_low, log_mass_high]
#     # def add_sanders_fmr(log_mass):
#     #     metals = compute_metals(log_mass, fm_s)
#     #     ax_metal_sfr.plot(fm_s, metals, ls='--', color='black', marker='None')
#     # for mass in log_masses:  
#     #     add_sanders_fmr(mass)



#     for ax in [ax_av_sfr, ax_av_metal, ax_stellarav_metal, ax_stellarav_sfr]:
#         scale_aspect(ax)
#         # ax.legend(fontsize=full_page_axisfont-4)
#     fig.savefig(imd.sed_paper_figures_dir + '/dust_metal_sfr.pdf', bbox_inches='tight')
#     plt.close('all')


def make_AV_panel_fig():
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, left=0.11, right=0.96, bottom=0.12, wspace=0.28,width_ratios=[1,1,1])
    ax_av_mass = fig.add_subplot(gs[0, 0])
    ax_balmer_mass = fig.add_subplot(gs[0, 1])
    ax_balmer_av_compare = fig.add_subplot(gs[0, 2])
    plot_a_vs_b_paper('median_log_mass', 'Prospector_AV_50', stellar_mass_label, prospector_dust2_label, 'None', axis_obj=ax_av_mass, yerr=True, plot_lims=[9, 11.5, -0.2, 2.5], fig=fig, use_color_df=True) 
    plot_a_vs_b_paper('median_log_mass', 'balmer_av_with_limit', stellar_mass_label, balmer_av_label, 'None', axis_obj=ax_balmer_mass, yerr=True, plot_lims=[9, 11.5, -0.2, 5], fig=fig, use_color_df=True, lower_limit=True) 
    regress_res = find_best_fit('median_log_mass', 'balmer_av_with_limit', exclude_limit=True)
    x_regress = np.arange(9, 11.8, 0.1)
    ax_balmer_mass.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='gray', label=f'Linear Fit', ls='--')
    print(f'Best fit to nebular av vs mass: slope {regress_res.slope}, yint {regress_res.intercept}')
    # Shapley's data
    mosdef_data_mass = np.array([9.252764612954188, 9.73301737756714, 10.0173775671406, 10.437598736176936]) #Shapley 2022
    mosdef_data_decs = np.array([3.337349397590363, 3.4548192771084363, 3.7801204819277103, 4.512048192771086])
    mosdef_data_balmeravs = compute_balmer_av(mosdef_data_decs)
    ax_balmer_mass.plot(mosdef_data_mass, mosdef_data_balmeravs, color='black', marker='s', ms=10, mec='black', ls='--', zorder=1000000, label='z=2.3 MOSDEF (Shapley+ 2022)')
    ax_balmer_mass.legend(fontsize=14)
    regress_res = find_best_fit('Prospector_AV_50', 'balmer_av_with_limit', exclude_limit=True)
    x_regress = np.arange(-0.2, 2.5, 0.1)
    print(f'Best fit to Nebular vs Stellar av: slope {regress_res.slope}, yint {regress_res.intercept}')
    ax_balmer_av_compare.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='black', label='Linear fit', ls='--')
    plot_a_vs_b_paper('Prospector_AV_50', 'balmer_av_with_limit', prospector_dust2_label, balmer_av_label, 'None', axis_obj=ax_balmer_av_compare, yerr=True, plot_lims=[-0.2, 2, -0.2, 5], fig=fig, use_color_df=True, prospector_xerr=True, one_to_one=False, factor_of_2=True, lower_limit=True)

    ax_balmer_av_compare.legend(fontsize=14, loc=2)
    print(f'Best fit to Av difference vs SFR: slope {regress_res.slope}, yint {regress_res.intercept}')


    for ax in [ax_av_mass, ax_balmer_mass, ax_balmer_av_compare]:
        scale_aspect(ax)
        # ax.legend(fontsize=full_page_axisfont-4)
    fig.savefig(imd.sed_paper_figures_dir + '/attenuation_panel.pdf', bbox_inches='tight')
    plt.close('all')

def make_av_comparison():
    av_difference_label = 'Nebular A$_V$ - Stellar A$_V$'
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, left=0.11, right=0.96, bottom=0.12, wspace=0.42, width_ratios=[1,1,1])
    ax_balmer_av_compare = fig.add_subplot(gs[0, 0])
    ax_avdiff_mass = fig.add_subplot(gs[0, 1])
    ax_avdiff_sfr = fig.add_subplot(gs[0, 2])
    def add_r(ax, regress, offset=0):
        ax.text(0.03, 0.93+offset, f"r={round(regress.rvalue, 2):.2f}", fontsize=18, transform = ax.transAxes)
    plot_a_vs_b_paper('median_log_mass', 'AV_difference_with_limit', stellar_mass_label, av_difference_label, 'None', axis_obj=ax_avdiff_mass, yerr=True, fig=fig, use_color_df=True, lower_limit=180, plot_lims=[9, 11.5, -1, 3])
    # plot_a_vs_b_paper('computed_log_ssfr_with_limit', 'AV_difference_with_limit', ssfr_label, av_difference_label, 'None', axis_obj=ax_avdiff_mass, yerr=True, fig=fig, use_color_df=True, lower_limit=180, plot_lims=[-10, -7, -1, 3])

    x_regress = np.arange(8, 12, 0.1)
    regress_res, points_16, points_84 = bootstrap_fit('median_log_mass', 'AV_difference_with_limit', x_regress, exclude_limit=True)
    add_r(ax_avdiff_mass, regress_res)
    ax_avdiff_mass.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='black', ls='--')
    ax_avdiff_mass.fill_between(x_regress, points_16, points_84, facecolor="gray", alpha=0.3)

    # ax_avdiff_mass.legend(fontsize=14, loc=2)
    plot_a_vs_b_paper('computed_log_sfr_with_limit', 'AV_difference_with_limit', sfr_label, av_difference_label, 'None', axis_obj=ax_avdiff_sfr, yerr=True, fig=fig, use_color_df=True, lower_limit=135, plot_lims=[0, 2, -1, 3])
    x_regress = np.arange(-1, 3, 0.1)
    regress_res, points_16, points_84 = bootstrap_fit('computed_log_sfr_with_limit', 'AV_difference_with_limit', x_regress, exclude_limit=True)
    # ax_balmer_mass.plot(x_regress, yints[1] + slopes[1]*x_regress, color='green', ls='-')
    # ax_balmer_mass.plot(x_regress, yints[2] + slopes[2]*x_regress, color='green', ls='-')
    ax_avdiff_sfr.fill_between(x_regress, points_16, points_84, facecolor="gray", alpha=0.3)
    add_r(ax_avdiff_sfr, regress_res)
    ax_avdiff_sfr.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='black', ls='--')
    # ax_avdiff_sfr.legend(fontsize=14, loc=2)

    plot_a_vs_b_paper('Prospector_AV_50', 'balmer_av_with_limit', prospector_dust2_label, balmer_av_label, 'None', axis_obj=ax_balmer_av_compare, yerr=True, plot_lims=[-0.2, 2.5, -0.2, 5], fig=fig, use_color_df=True, prospector_xerr=True, one_to_one=True, factor_of_2=False, lower_limit=180)
    # regress_res = find_best_fit('Prospector_AV_50', 'balmer_av_with_limit', exclude_limit=True)
    x_regress = np.arange(-0.2, 3.0, 0.1)
    regress_res, points_16, points_84 = bootstrap_fit('Prospector_AV_50', 'balmer_av_with_limit', x_regress, exclude_limit=True)
    # ax_balmer_mass.plot(x_regress, yints[1] + slopes[1]*x_regress, color='green', ls='-')
    # ax_balmer_mass.plot(x_regress, yints[2] + slopes[2]*x_regress, color='green', ls='-')
    ax_balmer_av_compare.fill_between(x_regress, points_16, points_84, facecolor="gray", alpha=0.3)
    print(f'Best fit to Nebular vs Stellar av: slope {regress_res.slope}, yint {regress_res.intercept}')
    ax_balmer_av_compare.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='black', ls='--')
    add_r(ax_balmer_av_compare, regress_res, offset=-0.12)
    ax_balmer_av_compare.plot([-100, -100], [-100, -100], color='red', ls='--', label='one-to-one')
    ax_balmer_av_compare.legend(fontsize=14, loc=2)
    print(f'Best fit to Av difference vs SFR: slope {regress_res.slope}, yint {regress_res.intercept}')


    for ax in [ax_balmer_av_compare, ax_avdiff_mass, ax_avdiff_sfr]:
        scale_aspect(ax)
        # ax.legend(fontsize=full_page_axisfont-4)
    fig.savefig(imd.sed_paper_figures_dir + '/av_comparison.pdf', bbox_inches='tight')
    plt.close('all')


# def make_av_difference_fig():
#     av_difference_label = 'Nebular A$_V$ - Stellar A$_V$'
#     fig = plt.figure(figsize=(11, 7))
#     gs = GridSpec(1, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.42, width_ratios=[1,1])
#     ax_avdiff_mass = fig.add_subplot(gs[0, 0])
#     ax_avdiff_sfr = fig.add_subplot(gs[0, 1])
#     plot_a_vs_b_paper('median_log_mass', 'AV_difference_with_limit', stellar_mass_label, av_difference_label, 'None', axis_obj=ax_avdiff_mass, yerr=True, fig=fig, use_color_df=True, lower_limit=True, plot_lims=[9, 11.5, -1, 3])
#     regress_res = find_best_fit('median_log_mass', 'AV_difference_with_limit', exclude_limit=True)
#     x_regress = np.arange(8, 12, 0.1)
#     ax_avdiff_mass.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='black', label='Linear fit', ls='--')
#     ax_avdiff_mass.legend(fontsize=14, loc=2)
#     plot_a_vs_b_paper('computed_log_sfr_with_limit', 'AV_difference_with_limit', sfr_label, av_difference_label, 'None', axis_obj=ax_avdiff_sfr, yerr=True, fig=fig, use_color_df=True, lower_limit=True, plot_lims=[0, 2, -1, 3])
#     regress_res = find_best_fit('computed_log_sfr_with_limit', 'AV_difference_with_limit', exclude_limit=True)
#     x_regress = np.arange(-1, 3, 0.1)
#     ax_avdiff_sfr.plot(x_regress, regress_res.intercept + regress_res.slope*x_regress, color='black', label='Linear fit', ls='--')
#     ax_avdiff_sfr.legend(fontsize=14, loc=2)
#     for ax in [ax_avdiff_mass, ax_avdiff_sfr]:
#         scale_aspect(ax)
#         # ax.legend(fontsize=full_page_axisfont-4)
#     fig.savefig(imd.sed_paper_figures_dir + '/av_difference.pdf', bbox_inches='tight')
#     plt.close('all')

def make_AV_panel_fig_old():
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

def bootstrap_fit(x_col, y_col, xpoints, exclude_limit=True, bootstrap=200):
    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    if exclude_limit == True:
        cluster_summary_df = cluster_summary_df[cluster_summary_df['flag_hb_limit']==0]
    xvals = cluster_summary_df[x_col]
    yvals = cluster_summary_df[y_col]
    if y_col == 'Prospector_AV_50':
        yerr_low = cluster_summary_df['Prospector_AV_16']
        yerr_high = cluster_summary_df['Prospector_AV_84']
    else:
        yerr_low = cluster_summary_df['err_' + y_col + '_low']
        yerr_high = cluster_summary_df['err_' + y_col + '_high']
    boot_slopes = []
    boot_yints = []
    for boot in range(bootstrap):
        new_ys = [draw_asymettric_error(yvals.iloc[i], yerr_low.iloc[i], yerr_high.iloc[i]) for i in range(len(yvals))]
        regress_res = linregress(xvals, new_ys)
        boot_slopes.append(regress_res.slope)
        boot_yints.append(regress_res.intercept)
    all_points = [boot_yints[i] + boot_slopes[i]*xpoints for i in range(len(boot_slopes))]
    

    
    def get_points(percentile):
        percentile_points = []
        for i in range(len(all_points[0])):
            point_js = [all_points[j][i] for j in range(len(all_points))]
            percentile_points.append(np.percentile(point_js, percentile))
        return percentile_points
    points_16 = get_points(16)
    points_84 = get_points(84)
    regress_res = linregress(xvals, yvals)
  
    return regress_res, points_16, points_84
    





def make_SFR_compare_fig():
    fig = plt.figure(figsize=(6.2, 6))
    gs = GridSpec(1, 1, left=0.11, right=0.96, bottom=0.12)
    ax_sfr = fig.add_subplot(gs[0, 0])

    plot_a_vs_b_paper('log_Prospector_ssfr50_multiplied_normalized', 'computed_log_sfr_with_limit', 'Prospector Normalized SED SFR', 'log$_{10}$(H$\\mathrm{\\alpha}$ SFR) (M$_\odot$ / yr)', 'None', axis_obj=ax_sfr, xerr=True, yerr=True, lower_limit=180, plot_lims=[-1.1, 2.1, 0.1, 2.1], fig=fig, one_to_one=True, use_color_df=True, add_numbers=False)
    ax_sfr.plot([-100, -100], [-100, -100], color='red', ls='--', label='one-to-one')
    ax_sfr.legend(fontsize=16, loc=2)
    ax_sfr.tick_params(labelsize=full_page_axisfont)

    from matplotlib.cm import ScalarMappable
    import matplotlib as mpl
    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=9.4, vmax=11.05) 
    sm =  ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax_sfr)
    cbar.set_label(stellar_mass_label, fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    scale_aspect(ax_sfr)
    fig.savefig(imd.sed_paper_figures_dir + '/sfr_compare_normalized.pdf', bbox_inches='tight')

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
    fig.savefig(imd.sed_paper_figures_dir + '/uvj_bpt.pdf', bbox_inches='tight')
    plt.close('all')
    
def make_sfr_mass_uvj_bpt_4panel(n_clusters=20, snr_thresh=2):
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(2, 2, left=0.11, right=0.96, bottom=0.12, wspace=0.28, height_ratios=[1,1],width_ratios=[1,1])
    ax_ssfr = fig.add_subplot(gs[0, 1])
    ax_metallicity = fig.add_subplot(gs[1, 1])
    ax_uvj = fig.add_subplot(gs[0, 0])
    ax_bpt = fig.add_subplot(gs[1, 0])
    #SFR/Metallicity
    plot_a_vs_b_paper('median_log_mass', 'computed_log_ssfr_with_limit', stellar_mass_label, ssfr_label, 'None', axis_obj=ax_ssfr, yerr=True, plot_lims=[9, 11.5, -10.8, -7.5], lower_limit=180, fig=fig, use_color_df=True) #, color_var='median_U_V'
    plot_a_vs_b_paper('median_log_mass', 'O3N2_metallicity_upper_limit', stellar_mass_label, metallicity_label, 'None', axis_obj=ax_metallicity, yerr=True, plot_lims=[9, 11.5, 8.15, 9.17], fig=fig, lower_limit=360)
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

    ax_bpt.legend(fontsize=14)

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