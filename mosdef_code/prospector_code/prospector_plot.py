'''Funcitons for plotting after prospector has run'''

import os
import pickle
import numpy as np
from astropy.io import ascii
from uvj_clusters import plot_uvj_cluster
from emission_measurements import read_emission_df
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from bpt_clusters import plot_bpt
import cluster_data_funcs as cdf
from emission_measurements import read_emission_df
from plot_mass_sfr import plot_mass_sfr_cluster, read_sfr_df, get_all_sfrs_masses
import initialize_mosdef_dirs as imd


def load_obj(name, run_name):
    with open(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs' + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def make_plots(groupID, run_name, mask=False, savename='False'):
    """Plots the observations vs the sps model

    Parameters:
    groupID (int): Number of the group to plot
    run_name (str): Name of the current run, used to sort folders
    mask(boolean): Set to True if there was a mask
    savename (str): Set to the name you want to save the file under

    """
    # res = load_obj(f'{groupID}_res')
    obs = load_obj(f'{groupID}_obs', run_name)

    spec_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs' + 
                         f'/{groupID}_spec.csv').to_pandas()
    phot_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs' +
                         f'/{groupID}_phot.csv').to_pandas()
    lines_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs' +
                          f'/{groupID}_lines.csv').to_pandas()

    save_dir = imd.prospector_plot_dir + f'/{run_name}_plots/'
    if savename == 'False':
        savename = f'group{groupID}'

    # Read in nebular emission lines from mosdef
    mosdef_lines_df = ascii.read(imd.loc_mosdef_elines).to_pandas()
        

    # Figure setup
    fig = plt.figure(figsize=(14, 8))
    ax_main = fig.add_axes([0.04, 0.40, 0.44, 0.57])
    ax_zoomHb = fig.add_axes([0.63, 0.73, 0.20, 0.24])
    ax_zoomHa = fig.add_axes([0.88, 0.73, 0.10, 0.24])
    spectra_axes = [ax_main, ax_zoomHa, ax_zoomHb]

    # Plot the UVJ diagram
    ax_UVJ = fig.add_axes([0.76, 0.08, 0.20, 0.25])
    plot_uvj_cluster(groupID, ax_UVJ)
    ax_UVJ.set_xlabel('V-J')
    ax_UVJ.set_ylabel('U-V')

    # Setup the BPT Diagram
    ax_BPT = fig.add_axes([0.76, 0.38, 0.20, 0.25])

    # Plot the SFR/Mass Diagram
    ax_SFR = fig.add_axes([0.52, 0.08, 0.20, 0.25])
    sfr_df = read_sfr_df()
    all_sfrs_res = get_all_sfrs_masses(sfr_df)
    plot_mass_sfr_cluster(groupID, all_sfrs_res, axis_obj=ax_SFR)
    ax_SFR.set_xlabel('log(Stellar Mass) (M_sun)')
    ax_SFR.set_ylabel('log(SFR) (M_sun/yr)')

    # Axis for residuals between model and observations
    ax_residual = fig.add_axes([0.04, 0.08, 0.44, 0.25])

    # Set the wavelength limits
    start_spec = phot_df['rest_wavelength'].iloc[0]
    end_spec = phot_df['rest_wavelength'].iloc[-1]
    spec_idxs = np.logical_and(
        spec_df['rest_wavelength'] > start_spec, spec_df['rest_wavelength'] < end_spec)

    # Plot the BPT diagram:
    bpt_lines = [6585, 6563, 5008, 4861]
    bpt_names = ['NII', 'Ha', 'OIII', 'Hb']
    bpt_fluxes = []
    bpt_errs_pct = []
    # Compute the BPT value for our composite
    for bpt_line in bpt_lines:
        # Find the nearest line in cloudy:
        idx_nearest = np.argmin(np.abs(lines_df['rest_wavelength'] - bpt_line))
        line_wave = lines_df['rest_wavelength'].iloc[idx_nearest]
        line_flux = lines_df['lines50'].iloc[idx_nearest]
        line_errs_pct = (((line_flux - lines_df['lines16'].iloc[idx_nearest]) / line_flux),
                         ((lines_df['lines84'].iloc[idx_nearest] - line_flux) / line_flux))
        bpt_fluxes.append(line_flux)
        bpt_errs_pct.append(line_errs_pct)
    # Calculate the bpt points
    bpt_x_rat = np.log10(bpt_fluxes[0] / bpt_fluxes[1])
    err_bpt_x_rat = (0.434 * np.sqrt(bpt_errs_pct[0][0]**2 + bpt_errs_pct[1][
                     0]**2), 0.434 * np.sqrt(bpt_errs_pct[0][1]**2 + bpt_errs_pct[1][1]**2))
    bpt_y_rat = np.log10(bpt_fluxes[2] / bpt_fluxes[3])
    err_bpt_y_rat = (0.434 * np.sqrt(bpt_errs_pct[2][0]**2 + bpt_errs_pct[3][
                     0]**2), 0.434 * np.sqrt(bpt_errs_pct[2][1]**2 + bpt_errs_pct[3][1]**2))
    composite_bpt_point = [bpt_x_rat, bpt_y_rat]
    composite_bpt_errs = [err_bpt_x_rat, err_bpt_y_rat]
    # Read in the emission for cluster galaxies
    emission_df = read_emission_df()
    # Find the names and ids of cluster galaxies
    cluster_names, fields_ids = cdf.get_cluster_fields_ids(groupID)
    fields_ids = [(obj[0], int(obj[1])) for obj in fields_ids]
    plot_bpt(emission_df, fields_ids, axis_obj=ax_BPT,
             composite_bpt_point=composite_bpt_point, composite_bpt_errs=composite_bpt_errs)
    ax_BPT.set_xlabel('log(N[II] 6583 / H$\\alpha$)')
    ax_BPT.set_ylabel('log(O[III] 5007 / H$\\beta$)')

    # Find the ranges for Ha and Hb
    Ha_range = [6500, 6625]
    Hb_range = [4800, 5050]
    Ha_idxs = np.logical_and(spec_df['rest_wavelength'] > Ha_range[
                             0], spec_df['rest_wavelength'] < Ha_range[1])
    Hb_idxs = np.logical_and(spec_df['rest_wavelength'] > Hb_range[
                             0], spec_df['rest_wavelength'] < Hb_range[1])

    ax_ranges = [[start_spec, end_spec], Ha_range, Hb_range]

    for j in range(len(spectra_axes)):
        ax = spectra_axes[j]
        ax_range = ax_ranges[j]
        # Plot photometry
        ax.errorbar(phot_df['rest_wavelength'], phot_df['rest_wavelength'] * obs['f_lambda'], color='black', yerr=phot_df['rest_wavelength'] * obs[
                    'err_f_lambda'], ls='-', marker='o', label='Observations', zorder=1)

        y_model = np.array(phot_df['rest_wavelength']
                           * phot_df['phot50_flambda'])
        y_model_16 = phot_df['rest_wavelength'] * phot_df['phot16_flambda']
        y_model_84 = phot_df['rest_wavelength'] * phot_df['phot84_flambda']
        model_errs = np.vstack((y_model - y_model_16, y_model_84 - y_model))
        ax.errorbar(np.array(phot_df['rest_wavelength']), y_model,
                    ls='-', marker='o', yerr=model_errs, color='blue', label='Model')

        # Plot spectrum
        ax.plot(spec_df['rest_wavelength'][spec_idxs], spec_df['rest_wavelength'][spec_idxs] * spec_df['spec50_flambda'][spec_idxs], '-',
                color='orange', label='Model spectrum', zorder=3)

        # Plot Emission line and labels - plots all from CLOUDY
        # for i in range(len(line_waves)):
        #     if np.logical_and(line_waves[i] > z0_spec_wavelength[0], line_waves[i] < z0_spec_wavelength[-1]):
        #         ax.axvline(line_waves[i],
        #                    ls='--', color='mediumseagreen')

        # Plot Emission line and labels - plots only those form mosdef
        text_height = 0.96
        for i in range(len(mosdef_lines_df)):
            # Get rest wavelength
            line_rest_wave = int(mosdef_lines_df.iloc[i]['Wavelength'])

            # Check if the line is in the range - if it's out, loop
            if np.logical_or(line_rest_wave < ax_range[0], line_rest_wave > ax_range[1]):
                continue

            # Find the nearest line in cloudy:
            idx_nearest = np.argmin(
                np.abs(lines_df['rest_wavelength'] - line_rest_wave))
            line_wave = lines_df['rest_wavelength'].iloc[idx_nearest]
            line_flux = lines_df['lines50'].iloc[idx_nearest]
            # Plot a green line where it is
            ax.axvline(line_wave, ls='--', color='mediumseagreen')

            # Add a label
            trans = transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.text(line_wave, text_height, mosdef_lines_df.iloc[
                    i]['Name'], transform=trans)
            text_height = text_height - 0.025

        ax.set_ylabel("$\lambda$ F$_\lambda$")
        ax.set_xlabel("Wavelength ($\AA$)")

    # Plot residuals
    ax_residual.plot(phot_df['rest_wavelength'], (phot_df['rest_wavelength'] * obs['f_lambda']) - (phot_df['rest_wavelength'] * phot_df['phot50_flambda']),
                     '-o', color='black')
    ax_residual.plot(phot_df['rest_wavelength'], np.zeros(
        len(phot_df['rest_wavelength'])), '--', color='grey')
    ax_residual.set_ylabel("$\lambda$ F$_\lambda$")
    ax_residual.set_xlabel("Wavelength ($\AA$)")
    ax_residual.set_xscale('log')

    # for i in range(len(lines_df)):
    #     if np.logical_and(lines_df.iloc[i]['Wavelength'] > spec_wavelength[0], lines_df.iloc[i]['Wavelength'] < spec_wavelength[-1]):
    #         ax.axvline(lines_df.iloc[i]['Wavelength'],
    #                    ls='--', color='mediumseagreen')
    ax_main.set_xscale('log')
    # ax_main.set_ylim(0.8 * np.percentile(z0_spec_wavelength[spec_idxs] * spectrum, 1),
    # 1.1 * np.percentile(z0_spec_wavelength[spec_idxs] * spectrum, 99))
    ax_main.set_ylim(0.8 * np.percentile(spec_df['rest_wavelength'][spec_idxs] * spec_df['spec50_flambda'][spec_idxs], 1),
                     1.1 * np.percentile(spec_df['rest_wavelength'][spec_idxs] * spec_df['spec50_flambda'][spec_idxs], 99))
    ax_main.set_xlim(spec_df['rest_wavelength'][spec_idxs].iloc[0] -
                     30, spec_df['rest_wavelength'][spec_idxs].iloc[-1] + 3000)

    ax_zoomHa.set_xlim(Ha_range[0], Ha_range[1])
    ax_zoomHb.set_xlim(Hb_range[0], Hb_range[1])
    # ax_zoomHa.set_ylim(
    #     0.8 * np.percentile(z0_spec_wavelength[Ha_idxs] * spectrum[Ha_idxs], 1), 1.1 * np.percentile(z0_spec_wavelength[Ha_idxs] * spectrum[Ha_idxs], 99))
    # ax_zoomHb.set_ylim(
    # 0.8 * np.percentile(z0_spec_wavelength[Hb_idxs] * spectrum[Hb_idxs], 1),
    # 1.1 * np.percentile(z0_spec_wavelength[Hb_idxs] * spectrum[Hb_idxs],
    # 99))
    ax_zoomHa.set_ylim(
        0.8 * np.percentile(spec_df['rest_wavelength'][Ha_idxs] * spec_df['spec50_flambda'][Ha_idxs], 1), 1.1 * np.percentile(spec_df['rest_wavelength'][Ha_idxs] * spec_df['spec50_flambda'][Ha_idxs], 99))
    ax_zoomHb.set_ylim(
        0.8 * np.percentile(spec_df['rest_wavelength'][Hb_idxs] * spec_df['spec50_flambda'][Hb_idxs], 1), 1.1 * np.percentile(spec_df['rest_wavelength'][Hb_idxs] * spec_df['spec50_flambda'][Hb_idxs], 99))

    # Mask for halpha
    if mask == True:
        ax_main.axvspan(4500, 5300, facecolor='r', alpha=0.5, label="Mask")
        ax_main.axvspan(6100, 6900, facecolor='r', alpha=0.5)

    # Save the plot
    ax_main.legend()
    fig.savefig(save_dir + '/' + savename + '_fit.pdf')
    plt.close('all')


def make_all_prospector_plots(n_clusters, run_name):
    '''Makes the plots from the outputs of the prospector run on Savio
    
    n_clusters (int): Number of composite clusters
    run_name (str): Name of the current run, used to sort folders


    '''
    for groupID in range(n_clusters):
        if os.path.exists(imd.prospector_fit_csvs_dir + f'/{groupID}_phot.csv'):
            print(f'Making plot for group {groupID}')
            make_plots(groupID, run_name)



