from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy import units as u
from astropy.nddata import Cutout2D
from uncover_read_data import read_supercat, read_segmap, read_bcg_surface_brightness
from uncover_make_sed import make_full_phot_sed
from uncover_prospector_seds import read_prospector, make_prospector
from full_phot_sample_selection import line_list
from sedpy import observate
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from glob import glob
import sys
import numpy as np
from plot_vals import scale_aspect
from matplotlib.colors import Normalize
from plot_log_linear_rgb import make_log_rgb
import matplotlib.patheffects as pe
from uncover_cosmo import find_pix_per_kpc, pixel_scale
import initialize_mosdef_dirs as imd
import shutil
import pandas as pd
from full_phot_make_prospector_models import prospector_abs_spec_folder, read_abs_sed
from uncover_sed_filters import unconver_read_filters
from full_phot_read_data import read_line_sample_df

phot_df_loc = '/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_linecoverage_ha_pab_paa.csv'
figure_save_loc = '/Users/brianlorenz/uncover/Figures/PHOT_sample/'

colors = ['red', 'green', 'blue']
cmap='inferno'



def calc_lineflux_and_linemap(id_dr3, line_name, phot_df, supercat_df, image_size=(100,100), snr_thresh=2, bcg_flag=0, make_linemap=False):
    """Given a DR3 id and a line, make the linemap
    
    Parameters:
    id_dr3 (int): DR3 id from UNCOVER
    line_name (str): Name of line in phot catalog, e.g. 'Halpha'
    phot_df (pd.Dataframe): Phot datafrme, from full_phot_sample_select
    supercat_df (pd.Dataframe): SUPER catalog from uncover
    """


    

    # Maybe set up dynamic image size scaling
    phot_df_row = phot_df[phot_df['id'] == id_dr3]
    supercat_row = supercat_df[supercat_df['id']==id_dr3]
    line_filters, line_images, wht_line_images, obj_segmap, line_photfnus = make_3color(id_dr3, line_name, phot_df_row, supercat_df, plot=False, image_size=image_size)
    line_sedpy_name = line_filters[1].replace('f', 'jwst_f')
    line_sedpy_filt = observate.load_filters([line_sedpy_name])[0]
    line_filter_width = line_sedpy_filt.rectangular_width
    
    redcont_filt = phot_df_row[f'{line_name}_filter_redcont'].iloc[0]
    obs_filt = phot_df_row[f'{line_name}_filter_obs'].iloc[0]
    bluecont_filt = phot_df_row[f'{line_name}_filter_bluecont'].iloc[0]
    filters = [redcont_filt, obs_filt, bluecont_filt]

    redshift = phot_df_row['z_50'].iloc[0]

    # cont_percentile, line_flux, boot_lines, sed_fluxes, wave_pct, line_rest_wavelength, cont_value = plot_sed_around_line(id_dr3, line_name, filters, redshift, bootstrap=1000)
    line_flux, sed_fluxes, line_wave_rest, cont_value, boot_lines, wave_pct, cont_percentile, offset_quality_factor, chi2, scaled_chi2 = plot_sed_around_line_prospector(id_dr3, line_name, filters, redshift, bootstrap=1000)

   
    err_lineflux_low = line_flux - np.percentile(boot_lines, 16)
    err_lineflux_high = np.percentile(boot_lines, 86) - line_flux
    flux_snr = line_flux / np.std(boot_lines)
    lineflux_info = [line_flux, err_lineflux_low, err_lineflux_high, flux_snr, cont_value, offset_quality_factor, chi2, scaled_chi2]

    subdir_str = '' # Will save to a different folder than the main one
    if flux_snr < snr_thresh:
        print(f'SNR < {snr_thresh}')
        subdir_str = '_low_snr/'
    if bcg_flag > 0:
        print(f'Too close to bcg')
        subdir_str = '_bcg_flag/'

    # Make linemaps - NEED TO REMAKE USING PROSPECTOR VALUES IF WE WANT TO DO THAT
    if make_linemap == True:
        # Need to multiply the image fluxes by 1e-8 to turn them from 10nJy to Jy
        jy_convert_factor = 1e-8
        # Get the data values
        line_red_image_data = jy_convert_factor*line_images[0].data
        line_green_image_data = jy_convert_factor*line_images[1].data
        line_blue_image_data = jy_convert_factor*line_images[2].data
        # Get the noise values
        line_red_image_noise = jy_convert_factor*(1/np.sqrt(wht_line_images[0].data))
        line_green_image_noise = jy_convert_factor*(1/np.sqrt(wht_line_images[1].data))
        line_blue_image_noise = jy_convert_factor*(1/np.sqrt(wht_line_images[2].data))
        line_image_noises = [line_red_image_noise, line_green_image_noise, line_blue_image_noise]
        
        linemap, contmap, err_linemap = compute_line(cont_percentile, line_red_image_data, line_green_image_data, line_blue_image_data, redshift, 0, line_filter_width, line_rest_wavelength, images=True, image_noises=line_image_noises, wave_pct=wave_pct)

        linemap_snr = linemap / err_linemap

        # Norm values
        cont_lower_pct = 10
        cont_upper_pct = 99.99
        cont_scalea = 1e30
        linemap_lower_pct = 10
        linemap_upper_pct = 99.9
        linemap_scalea = 150
        dustmap_lower_pct = 40
        dustmap_upper_pct = 90
        dustmap_scalea = 100

        contmap_logscaled = make_log_rgb(contmap, contmap, contmap, scalea=cont_scalea)[:,:,0]
        linemap_logscaled = make_log_rgb(linemap, linemap, linemap, scalea=linemap_scalea)[:,:,0]

        contmap_norm  = get_norm(contmap_logscaled, lower_pct=cont_lower_pct, upper_pct=cont_upper_pct)
        linemap_norm = get_norm(linemap_logscaled, lower_pct=linemap_lower_pct, upper_pct=linemap_upper_pct)


        fig, axarr = plt.subplots(1,3,figsize=(12,4))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        ax_image = axarr[0]
        ax_contmap = axarr[1]
        ax_linemap = axarr[2]

        plot_single_3color(ax_image, line_images, filters, showtext=False)
        ax_contmap.imshow(contmap_logscaled, cmap=cmap, norm=contmap_norm)
        ax_linemap.imshow(linemap_logscaled, cmap=cmap, norm=linemap_norm)

        text_height = 0.92
        text_start = 0.01
        text_sep = 0.37
        ax_image.text(text_start, text_height, f'{filters[2][2:].upper()}', fontsize=14, transform=ax_image.transAxes, color='blue', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax_image.text(text_start+text_sep, text_height, f'{filters[1][2:].upper()}', fontsize=14, transform=ax_image.transAxes, color='green', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax_image.text(text_start+2*text_sep, text_height, f'{filters[0][2:].upper()}', fontsize=14, transform=ax_image.transAxes, color='red', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax_image.text(0.80, 0.04, f'{id_dr3}', fontsize=10, transform=ax_image.transAxes, color='white')
        axis_x = 0.05
        axis_y = 0.05
        axis_to_data = ax_image.transAxes + ax_image.transData.inverted()
        data_x, data_y = axis_to_data.transform((axis_x, axis_y))
        data_x2, data_y2 = axis_to_data.transform((axis_x, axis_y+0.02))
        ax_image.plot([data_x,data_x+(0.5/pixel_scale)], [data_y,data_y], ls='-', color='white', lw=3)
        # ax_ha_image_paper.text(5, 9, '1kpc', color='white')
        ax_image.text(data_x, data_y2, '0.5"', color='white')


        ax_contmap.text(text_start, text_height, f'Continuum', fontsize=14, transform=ax_contmap.transAxes, color='white', path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        ax_linemap.text(text_start, text_height, f'{line_name} map', fontsize=14, transform=ax_linemap.transAxes, color='white', path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        ax_linemap.text(1-text_start, text_height, f'z={redshift:0.2f}', fontsize=14, transform=ax_linemap.transAxes, color='white', horizontalalignment='right', path_effects=[pe.withStroke(linewidth=3, foreground="black")])


        for ax in axarr:
            scale_aspect(ax)
            ax.set_xticks([]); ax.set_yticks([])

        imd.check_and_make_dir(figure_save_loc)
        imd.check_and_make_dir(figure_save_loc+'linemaps/')
        imd.check_and_make_dir(figure_save_loc+f'linemaps/{line_name}_linemaps/')
        imd.check_and_make_dir(figure_save_loc+f'linemaps/{line_name}_linemaps{subdir_str}/')
        fig.savefig(figure_save_loc+f'linemaps/{line_name}_linemaps{subdir_str}/{id_dr3}_{line_name}_linemap.pdf', bbox_inches='tight')
        plt.close('all')
    return lineflux_info, subdir_str

def get_norm(image_map, scalea=1, lower_pct=10, upper_pct=99):
        imagemap_gt0 = image_map[image_map>0.0001]

        # norm = LogNorm(vmin=np.percentile(imagemap_gt0,lower_pct), vmax=np.percentile(imagemap_gt0,upper_pct))
        norm = Normalize(vmin=np.percentile(imagemap_gt0,lower_pct), vmax=np.percentile(imagemap_gt0,upper_pct))
        return norm
""" old function without using prospector
def plot_sed_around_line(id_dr3, line_name, filters, redshift, bootstrap=1000):
    line_wave_rest = [line[1] for line in line_list if line[0]==line_name][0] # Angstrom
    line_wave_obs = (line_wave_rest * (1+redshift))/1e4 # micron

    sed_df = make_full_phot_sed(id_dr3)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.axvline(line_wave_obs, ls='--', color='green') # observed line
    
    # def set_error_floor(flux, err_flux, floor_pct=0.05):
    #     err_floor = flux*0.05
    #     if err_flux < err_floor:
    #         err_flux = err_floor
    #     return err_flux

    # Plot the 3 SED points
    for i in range(len(filters)):
        sed_row = sed_df[sed_df['filter'] == filters[i]]
        
        
        if i == 0:
            red_wave = sed_row['eff_wavelength'].iloc[0]
            red_flux = sed_row['flux'].iloc[0]
            err_red_flux = sed_row['err_flux'].iloc[0]
            # err_red_flux = set_error_floor(red_flux, err_red_flux)
        if i == 1:
            green_wave = sed_row['eff_wavelength'].iloc[0]
            green_flux = sed_row['flux'].iloc[0]
            err_green_flux = sed_row['err_flux'].iloc[0]
            # err_green_flux = set_error_floor(green_flux, err_green_flux)
        if i == 2:
            blue_wave = sed_row['eff_wavelength'].iloc[0]
            blue_flux = sed_row['flux'].iloc[0]
            err_blue_flux = sed_row['err_flux'].iloc[0]
            # err_blue_flux = set_error_floor(blue_flux, err_blue_flux)

        # Read and plot each filter curve
        sedpy_name = filters[i].replace('f_', 'jwst_')
        sedpy_filt = observate.load_filters([sedpy_name])[0]
        ax.plot(sedpy_filt.wavelength/1e4, sedpy_filt.transmission/6e5, ls='-', marker='None', color=colors[i], lw=1)
    
    
    ax.errorbar(red_wave, red_flux, yerr = err_red_flux, color='red', marker='o')
    ax.errorbar(green_wave, green_flux, yerr = err_green_flux, color='green', marker='o')
    ax.errorbar(blue_wave, blue_flux, yerr = err_blue_flux, color='blue', marker='o')


    # Compute the percentile to use when combining the continuum
    connect_color = 'purple'
    
    wave_pct = compute_wavelength_pct(blue_wave, green_wave, red_wave)
    cont_percentile = compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux)
    sedpy_name = filters[1].replace('f_', 'jwst_')
    sedpy_line_filt = observate.load_filters([sedpy_name])[0]
    filter_width = sedpy_line_filt.rectangular_width
    line_flux, cont_value = compute_line(cont_percentile, red_flux, green_flux, blue_flux, redshift, 0, filter_width, line_wave_rest)

    boot_lines = []
    if bootstrap > 0:
        for i in range(bootstrap):
            boot_red_flux = np.random.normal(loc=red_flux, scale=err_red_flux, size=1)
            boot_green_flux = np.random.normal(loc=green_flux, scale=err_green_flux, size=1)
            boot_blue_flux = np.random.normal(loc=blue_flux, scale=err_blue_flux, size=1)
            boot_cont_percentile = compute_cont_pct(blue_wave, green_wave, red_wave, boot_blue_flux, boot_red_flux)
            boot_line, boot_cont = compute_line(boot_cont_percentile, boot_red_flux[0], boot_green_flux[0], boot_blue_flux[0], redshift, 0, filter_width, line_wave_rest)            
            boot_line = boot_line 
            
            boot_lines.append(boot_line)
    boot_lines = np.array(boot_lines)


    ax.plot([red_wave, blue_wave], [red_flux, blue_flux], marker='None', ls='--', color=connect_color)
    ax.plot(green_wave, cont_value, marker='o', ls='None', color=connect_color)
    ax.plot([green_wave,green_wave], [green_flux, cont_value], marker='None', ls='-', color='green', lw=2)
        
       
    # Plot cleanup
    ax.set_xlabel('Wavelength (um)', fontsize=14)
    ax.set_ylabel('Flux (Jy)', fontsize=14)
    ax.tick_params(labelsize=14)
    # ax.set_xlim(0.8*line_wave_obs, 1.2*line_wave_obs)
    # ax.set_ylim(0, 1.2*np.max(spec_df['flux_calibrated_jy']))
    sed_fluxes = [red_flux, green_flux, blue_flux]

    imd.check_and_make_dir(figure_save_loc + f'sed_images/')
    imd.check_and_make_dir(figure_save_loc + f'sed_images/{line_name}_sed_images')
    fig.savefig(figure_save_loc + f'sed_images/{line_name}_sed_images/{id_dr3}_{line_name}_sed.pdf')

    return cont_percentile, line_flux, boot_lines, sed_fluxes, wave_pct, line_wave_rest, cont_value
"""

def plot_sed_around_line_prospector(id_dr3, line_name, filters, redshift, bootstrap=1000):

    # Use typical method with prospector and comtinuum to get line flux
    line_wave_rest = [line[1] for line in line_list if line[0]==line_name][0] # Angstrom
    line_wave_obs = (line_wave_rest * (1+redshift))/1e4 # micron

    sed_df = make_full_phot_sed(id_dr3)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.axvline(line_wave_obs, ls='--', color='green') # observed line
    
    # Plot the 3 SED points
    for i in range(len(filters)):
        sed_row = sed_df[sed_df['filter'] == filters[i]]
    
        if i == 0:
            red_wave = sed_row['eff_wavelength'].iloc[0]
            red_flux = sed_row['flux'].iloc[0]
            err_red_flux = sed_row['err_flux'].iloc[0]
        if i == 1:
            green_wave = sed_row['eff_wavelength'].iloc[0]
            green_flux = sed_row['flux'].iloc[0]
            err_green_flux = sed_row['err_flux'].iloc[0]
        if i == 2:
            blue_wave = sed_row['eff_wavelength'].iloc[0]
            blue_flux = sed_row['flux'].iloc[0]
            err_blue_flux = sed_row['err_flux'].iloc[0]

        # Read and plot each filter curve
        sedpy_name = filters[i].replace('f_', 'jwst_')
        sedpy_filt = observate.load_filters([sedpy_name])[0]
        ax.plot(sedpy_filt.wavelength/1e4, sedpy_filt.transmission/6e5, ls='-', marker='None', color=colors[i], lw=1)

    
    # Read in prospector model
    prospector_no_neb_df = ascii.read(f'{prospector_abs_spec_folder}{id_dr3}_prospector_no_neb.csv').to_pandas()
    prospector_abs_df = read_abs_sed(id_dr3)
    filters_jwstnames = [filtname.replace('f_', 'jwst_') for filtname in filters]
    
    prospector_red_cont_flux = prospector_abs_df[prospector_abs_df['filter_name'] == filters_jwstnames[0]]['obs_flux_jy'].iloc[0]
    prospector_green_cont_flux = prospector_abs_df[prospector_abs_df['filter_name'] == filters_jwstnames[1]]['obs_flux_jy'].iloc[0]
    prospector_blue_cont_flux = prospector_abs_df[prospector_abs_df['filter_name'] == filters_jwstnames[2]]['obs_flux_jy'].iloc[0]

    # Scale the prospector points to match the red/blue continuum points
    red_scale_factor = red_flux/prospector_red_cont_flux
    blue_scale_factor = blue_flux/prospector_blue_cont_flux
    total_scale_factor = np.mean([red_scale_factor, blue_scale_factor])

    prospector_red_cont_flux_scaled = prospector_red_cont_flux * total_scale_factor
    prospector_green_cont_flux_scaled = prospector_green_cont_flux * total_scale_factor
    prospector_blue_cont_flux_scaled = prospector_blue_cont_flux * total_scale_factor

    # Integrate Prospector using sedpy to get continuum point
    prospector_cont_flux = prospector_green_cont_flux_scaled # jy

    prospector_no_neb_df[f'rest_absorp_model_jy_scaled_to_{line_name}_cont'] = total_scale_factor*prospector_no_neb_df['rest_absorp_model_jy']
    prospector_no_neb_df.to_csv(f'{prospector_abs_spec_folder}{id_dr3}_prospector_no_neb.csv', index=False)

    ax.errorbar(red_wave, red_flux, yerr = err_red_flux, color='red', marker='o', zorder=10)
    ax.errorbar(green_wave, green_flux, yerr = err_green_flux, color='green', marker='o', zorder=10, label='Data')
    ax.errorbar(blue_wave, blue_flux, yerr = err_blue_flux, color='blue', marker='o', zorder=10)

    ax.plot(red_wave, prospector_red_cont_flux, color='red', marker='s', ls='None', zorder=10)
    ax.plot(green_wave, prospector_green_cont_flux, color='green', marker='s', ls='None', zorder=10, label='sedpy abs point')
    ax.plot(blue_wave, prospector_blue_cont_flux, color='blue', marker='s', ls='None', zorder=10)

    ax.step(prospector_no_neb_df['rest_wave']*(1+redshift)/10000, prospector_no_neb_df['rest_full_model_jy']/(1+redshift), color='orange', ls='-', marker='None', alpha=0.65, zorder=1, label='Emission model')
    ax.step(prospector_no_neb_df['rest_wave']*(1+redshift)/10000, prospector_no_neb_df['rest_absorp_model_jy']/(1+redshift), color='mediumseagreen', ls='-', marker='None', alpha=0.65, zorder=1, label='Absorption model')
    ax.step(prospector_no_neb_df['rest_wave']*(1+redshift)/10000, total_scale_factor*prospector_no_neb_df['rest_absorp_model_jy']/(1+redshift), color='magenta', ls='-', marker='None', alpha=0.65, zorder=1, label='Scaled abs model')
    
    ax.set_xlim(blue_wave-0.3, red_wave+0.3)
    ax.set_ylim(0, green_flux*1.1)
    ax.legend()

    # breakpoint()

    # Compute the percentile to use when combining the continuum
    connect_color = 'purple'
    
    # wave_pct = compute_wavelength_pct(blue_wave, green_wave, red_wave)
    # cont_percentile = compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux)
    sedpy_name = filters[1].replace('f_', 'jwst_')
    sedpy_line_filt = observate.load_filters([sedpy_name])[0]
    filter_width = sedpy_line_filt.rectangular_width
    line_flux, observed_wave = get_lineflux_from_cont(green_flux, prospector_cont_flux, line_wave_rest, redshift, filter_width)

    # Quality check - how close are the points to the continuum relative to the lineflux
    main_filter_offset_from_cont = np.abs(green_flux - prospector_cont_flux)
    red_offset_from_cont = np.abs(red_flux - prospector_red_cont_flux_scaled)
    blue_offset_from_cont = np.abs(blue_flux - prospector_blue_cont_flux_scaled)
    avg_cont_filter_offset = np.mean([red_offset_from_cont, blue_offset_from_cont])
    offset_quality_factor = main_filter_offset_from_cont / avg_cont_filter_offset # High quality is a good continuum fit

    # Compute chi2 for the continuum points
    observed_points = np.array([red_flux, blue_flux])
    expected_points = np.array([prospector_red_cont_flux_scaled, prospector_blue_cont_flux_scaled])
    chi2 = np.sum((observed_points - expected_points)**2 /expected_points)
    scaled_chi2 = chi2 / prospector_cont_flux


    ##### THINK about how to monte carlo - do it with prospector errors as well? Probably
    # Decided to just take original errors and apply them to the prospector points
    boot_lines = []
    if bootstrap > 0:
        for i in range(bootstrap):
            boot_red_flux = np.random.normal(loc=red_flux, scale=err_red_flux, size=1)
            boot_green_flux = np.random.normal(loc=green_flux, scale=err_green_flux, size=1)
            boot_blue_flux = np.random.normal(loc=blue_flux, scale=err_blue_flux, size=1)
            # Scale the prospector points to match the red/blue continuum points
            red_scale_factor_boot = red_flux/prospector_red_cont_flux
            blue_scale_factor_boot = blue_flux/prospector_blue_cont_flux
            total_scale_factor_boot = np.mean([red_scale_factor_boot, blue_scale_factor_boot])
            prospector_green_cont_flux_scaled_boot = prospector_green_cont_flux * total_scale_factor_boot
            boot_line_flux, observed_wave = get_lineflux_from_cont(boot_green_flux, prospector_green_cont_flux_scaled_boot, line_wave_rest, redshift, filter_width)
            boot_lines.append(boot_line_flux)
    boot_lines = np.array(boot_lines)


    # PLot the prospector point as well, instead of the normal purple point
    # ax.plot([red_wave, blue_wave], [red_flux, blue_flux], marker='None', ls='--', color=connect_color)
    ax.plot(green_wave, prospector_cont_flux, marker='s', ls='None', color=connect_color)
    ax.plot(red_wave, prospector_red_cont_flux_scaled, marker='s', ls='None', color='magenta')
    ax.plot(blue_wave, prospector_blue_cont_flux_scaled, marker='s', ls='None', color='magenta')
    ax.plot([green_wave,green_wave], [green_flux, prospector_cont_flux], marker='None', ls='-', color='green', lw=2)
       
    ax.text(0.03, 0.2, f'Flux: {line_flux:.3e}', transform=ax.transAxes)   
    
       
    # Plot cleanup
    ax.set_xlabel('Wavelength (um)', fontsize=14)
    ax.set_ylabel('Flux (Jy)', fontsize=14)
    ax.tick_params(labelsize=14)
    # ax.set_xlim(0.8*line_wave_obs, 1.2*line_wave_obs)
    # ax.set_ylim(0, 1.2*np.max(spec_df['flux_calibrated_jy']))
    sed_fluxes = [red_flux, green_flux, blue_flux]

    imd.check_and_make_dir(figure_save_loc + f'sed_images/')
    imd.check_and_make_dir(figure_save_loc + f'sed_images/{line_name}_sed_images_prospector_method/')
    fig.savefig(figure_save_loc + f'sed_images/{line_name}_sed_images_prospector_method/{id_dr3}_{line_name}_sed.pdf')

    wave_pct = compute_wavelength_pct(blue_wave, green_wave, red_wave)
    cont_percentile = compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux)

    return line_flux, sed_fluxes, line_wave_rest, prospector_cont_flux, boot_lines, wave_pct, cont_percentile, offset_quality_factor, chi2, scaled_chi2
    return cont_percentile, line_flux, boot_lines, sed_fluxes, wave_pct, line_wave_rest, cont_value


def make_3color(id_dr3, line_name, phot_df_row, supercat_df, plot = False, image_size=(100,100)): 
    obj_skycoord = get_coords(id_dr3, supercat_df)

    filt_blue = phot_df_row[f'{line_name}_filter_bluecont'].iloc[0].split('_')[1]
    filt_green = phot_df_row[f'{line_name}_filter_obs'].iloc[0].split('_')[1]
    filt_red = phot_df_row[f'{line_name}_filter_redcont'].iloc[0].split('_')[1]

    filters = [filt_red, filt_green, filt_blue]
   
    image_red, wht_image_red, photfnu_red = get_cutout(obj_skycoord, filt_red, size=image_size)
    image_green, wht_image_green, photfnu_green = get_cutout(obj_skycoord, filt_green, size=image_size)
    image_blue, wht_image_blue, photfnu_blue = get_cutout(obj_skycoord, filt_blue, size=image_size)
    images = [image_red, image_green, image_blue]
    wht_images = [wht_image_red, wht_image_green, wht_image_blue]
    photfnus = [photfnu_red, photfnu_green, photfnu_blue]

    obj_segmap = get_cutout_segmap(obj_skycoord, size=image_size)

    if plot == True:
        fig, ax = plt.subplots(figsize = (6,6))
        plot_single_3color(ax, images, filters)
        fig.savefig(figure_save_loc + f'three_colors/{id_dr3}_{line_name}_3color.pdf')
        plt.close('all')

    return filters, images, wht_images, obj_segmap, photfnus


# Plotting  single image
def plot_single_3color(ax, images, filters, showtext=True):
    image_red = images[0]
    image_green = images[1]
    image_blue = images[2]
    filt_red = filters[0]
    filt_green = filters[1]
    filt_blue = filters[2]
    image = make_lupton_rgb(image_red.data, image_green.data, image_blue.data, stretch=0.5)
    ax.imshow(image)
    text_height = 1.02
    text_start = 0.01
    text_sep = 0.2
    if showtext:
        ax.text(text_start, text_height, f'{filt_blue}', color='blue', fontsize=14, transform=ax.transAxes)
        ax.text(text_start+text_sep, text_height, f'{filt_green}', color='green', fontsize=14, transform=ax.transAxes)
        ax.text(text_start+2*text_sep, text_height, f'{filt_red}', color='red', fontsize=14, transform=ax.transAxes)
        ax.text(0.85, text_height, f'{line_name}', color='green', fontsize=14, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    
    

def get_coords(id_dr3, supercat_df):
    row = supercat_df[supercat_df['id']==id_dr3]
    obj_ra = row['ra'].iloc[0] * u.deg
    obj_dec = row['dec'].iloc[0] * u.deg
    obj_skycoord = SkyCoord(obj_ra, obj_dec)
    return obj_skycoord

def get_cutout(obj_skycoord, filt, size = (100, 100)):
    image, wht_image, wcs, wht_wcs, photfnu = load_image(filt)

    cutout = Cutout2D(image, obj_skycoord, size, wcs=wcs)
    wht_cutout = Cutout2D(wht_image, obj_skycoord, size, wcs=wht_wcs)
    return cutout, wht_cutout, photfnu

def get_cutout_segmap(obj_skycoord, size = (100, 100)):
    segmap, segmap_wcs = read_segmap()
    segmap_cutout = Cutout2D(segmap, obj_skycoord, size, wcs=segmap_wcs)
    return segmap_cutout

def load_image(filt):
    image_folder = '/Users/brianlorenz/uncover/Catalogs/psf_matched/'
    # image_str = f'uncover_v7.2_abell2744clu_{filt}_bcgs_sci_f444w-matched.fits'
    image_str = glob(image_folder + 'uncover_v7.*'+'*_abell2744clu_*'+filt+'*sci_f444w-matched.fits')
    wht_image_str = glob(image_folder + 'uncover_v7.*'+'*_abell2744clu_*'+filt+'*wht_f444w-matched.fits')
    if len(image_str) > 1:
        sys.exit(f'Error: multiple images found for filter {filt}')
    if len(image_str) < 1:
        sys.exit(f'Error: no image found for filter {filt}')
    image_str = image_str[0]
    wht_image_str = wht_image_str[0]
    with fits.open(image_str) as hdu:
        image = hdu[0].data
        wcs = WCS(hdu[0].header)
        photflam = hdu[0].header['PHOTFLAM']
        photplam = hdu[0].header['PHOTPLAM']
        photfnu = hdu[0].header['PHOTFNU']
    with fits.open(wht_image_str) as hdu_wht:
        wht_image = hdu_wht[0].data
        wht_wcs = WCS(hdu_wht[0].header)  
    return image, wht_image, wcs, wht_wcs, photfnu


# Moved this check into the catalog gnereation
# def check_medium_bands(line_name, phot_sample_df, supercat_df):
#     id_dr3_list = phot_sample_df[phot_sample_df[f'{line_name}_redshift_sigma'] > 4]['id'].to_list()
#     detected_ids = []
#     for id_dr3 in id_dr3_list:
#         phot_sample_row = phot_sample_df[phot_sample_df['id']==id_dr3]
#         supercat_row = supercat_df[supercat_df['id']==id_dr3]
        
#         obs_filt = phot_sample_row[f'{line_name}_filter_obs'].iloc[0]
#         redcont_filt = phot_sample_row[f'{line_name}_filter_redcont'].iloc[0]
#         bluecont_filt = phot_sample_row[f'{line_name}_filter_bluecont'].iloc[0]
        
#         if obs_filt == '-99' or redcont_filt == '-99' or bluecont_filt == '-99':
#             continue

#         null_obs = pd.isnull(supercat_row[obs_filt].iloc[0])
#         null_red = pd.isnull(supercat_row[redcont_filt].iloc[0])
#         null_blue = pd.isnull(supercat_row[bluecont_filt].iloc[0])
        
#         if null_obs + null_red + null_blue == 0:
#             detected_ids.append(id_dr3)
#     return detected_ids


def compute_wavelength_pct(blue_wave, green_wave, red_wave):
    total_wave_diff = red_wave - blue_wave
    line_wave_diff = green_wave - blue_wave
    wave_pct = line_wave_diff/total_wave_diff
    return wave_pct

def compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux):
    total_wave_diff = blue_wave - red_wave
    line_wave_diff = green_wave - red_wave
    cont_percentile = line_wave_diff/total_wave_diff
    if red_flux>blue_flux:
        cont_percentile = 1-cont_percentile
    return cont_percentile

def get_lineflux_from_cont(green_flx, cont_value, line_rest_wave, redshift, filter_width):
    line_value = green_flx - cont_value # Jy

    # Put in erg/s/cm2/Hz
    line_value = line_value * 1e-23
    
    # # Convert from f_nu to f_lambda
    c = 299792458 # m/s
    observed_wave = line_rest_wave * (1+redshift)
    line_value = line_value * ((c*1e10) / (observed_wave)**2) # erg/s/cm2/angstrom

    # # Multiply by filter width to just get F
    # Filter width is observed frame width
    line_value = line_value * filter_width  # erg/s/cm2
    return line_value, observed_wave


def compute_line(cont_pct, red_flx, green_flx, blue_flx, redshift, raw_transmission, filter_width, line_rest_wave, images=False, image_noises=[], wave_pct=50):
    """
    Fluxes in Jy
    Line rest wave in angstroms
    """
    if images == True:
        cont_value = np.percentile([red_flx, blue_flx], cont_pct*100, axis=0)
    else:
        cont_value = np.percentile([blue_flx, red_flx], cont_pct*100)

    line_value, observed_wave = get_lineflux_from_cont(green_flx, cont_value, line_rest_wave, redshift, filter_width)

    if images == True:
        err_cont_value = np.sqrt((((wave_pct)**2)*(image_noises[0])**2) + (((1-wave_pct)**2)*(image_noises[2])**2))
        err_line_value = np.sqrt(image_noises[1]**2 + err_cont_value**2)
        err_line_value = err_line_value * 1e-23
        err_line_value = err_line_value * ((c*1e10) / (observed_wave)**2)
        err_line_value = err_line_value * filter_width
        return line_value, cont_value, err_line_value

    return line_value, cont_value

def make_all_phot_linemaps(line_name):
    snr_thresh = 10
    bcg_thresh = 0.04

    phot_sample_df = ascii.read(phot_df_loc).to_pandas()
    bcg_df = read_bcg_surface_brightness()
    supercat_df = read_supercat()
    # RUNNING ONLY ON ONES THAT HAVE BOTH LINES FOR NOW
    # line_sample_df = read_line_sample_df(line_name)
    line_sample_df_ha_pab = read_line_sample_df('HalphaPaBeta')
    line_sample_df_paa_pab = read_line_sample_df('PaAlphaPaBeta')
    all_pab_df = pd.merge(line_sample_df_ha_pab, line_sample_df_paa_pab, how='outer')
    if line_name == 'Halpha':
        use_df = line_sample_df_ha_pab # Choosing to run halpha only if it's also detected with pabeta
    elif line_name == 'PaBeta':
        use_df = all_pab_df # running pabeta for anything with halpha or with paalpha
    elif line_name == 'PaAlpha':
        use_df = line_sample_df_paa_pab # Paalpha only with things htat have pabeta

    # paa_only_list = [26618, 28495, 30915, 37776, 39748, 41581, 45334, 51405, 54614, 54643, 56018, 61218]
    # paa_pab_spec_list = [26618, 28495, 30915, 37776, 54614, 54643, 61218]

    pandas_rows = []
    # for id_dr3 in paa_pab_spec_list:
    for id_dr3 in use_df['id'].to_list():
        phot_sample_row = phot_sample_df[phot_sample_df['id'] == id_dr3]
        redshift_sigma = phot_sample_row[f'{line_name}_redshift_sigma'].iloc[0]

        # Check the bcg flag
        if bcg_df[bcg_df['id_dr3'] == id_dr3]['bcg_surface_brightness'].iloc[0] > bcg_thresh:
            print(f'Too close to bcg')
            bcg_flag = 1
            # image_dir = '/Users/brianlorenz/uncover/Figures/PHOT_sample/first_run/'
            # shutil.copy(image_dir+f'linemaps_first/{id_dr3}_linemap.pdf', image_dir+f'bcg_flag/{id_dr3}_linemap.pdf')
        else:
            bcg_flag = 0

        print(f'Making {line_name} map for {id_dr3}')
        
        # full_gals_list = [17757, 17758, 30052, 30351, 32180, 32181, 36076, 37784, 40135, 46831, 47758, 48104, 49020, 49712, 49932, 50707, 51980, 54343, 59550, 64780, 13130, 22045, 23395, 29959, 30351, 32536, 33247, 33588, 33775, 35090, 40504, 40522, 43970, 46261, 46855, 47958, 54239, 54240, 54614, 54674, 55357, 55594, 57422, 60576, 60577, 60973, 64472, 64786, 67410]

        # if id_dr3 not in full_gals_list:
        #     continue
        #pandas row contains the lineflux
        pandas_row, subdir_str = calc_lineflux_and_linemap(id_dr3, line_name, phot_sample_df, supercat_df, snr_thresh=snr_thresh, bcg_flag=bcg_flag)
        pandas_row.insert(0, id_dr3)
        if subdir_str == '':
            subdir_str = 'no_flag'
            use_flag = 1
        else:
            use_flag = 0
        pandas_row.append(use_flag)
        pandas_row.append(subdir_str)
        pandas_row.append(redshift_sigma)
        pandas_rows.append(pandas_row)
    lineflux_df = pd.DataFrame(pandas_rows, columns=['id_dr3', f'{line_name}_flux', f'err_{line_name}_flux_low', f'err_{line_name}_flux_high', f'{line_name}_snr', f'{line_name}_cont_value', f'{line_name}_quality_factor', f'{line_name}_chi2', f'{line_name}_chi2_scaled', f'use_flag_{line_name}', f'flag_reason_{line_name}', f'{line_name}_redshift_sigma'])
    lineflux_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_lineflux_{line_name}.csv', index=False)

if __name__ == "__main__":
    phot_sample_df = ascii.read(phot_df_loc).to_pandas()
    bcg_df = read_bcg_surface_brightness()
    supercat_df = read_supercat()
    # ids = [40778, 45059, 53709, 12887, 13428, 25707, 23181, 49532]
    ids = [33853]
    for id_dr3 in ids:
        calc_lineflux_and_linemap(id_dr3, 'Halpha', phot_sample_df, supercat_df)
        calc_lineflux_and_linemap(id_dr3, 'PaBeta', phot_sample_df, supercat_df)

    # make_all_phot_linemaps('Halpha')
    # make_all_phot_linemaps('PaBeta')
    # make_all_phot_linemaps('PaAlpha')
    pass