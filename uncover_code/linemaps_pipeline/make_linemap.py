from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb
from sedpy import observate
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from glob import glob
import sys

# imports from other local files
from plot_log_linear_rgb import make_log_rgb
from uncover_input_data import line_coverage_path, uncover_image_folder, read_supercat, halpha_name, halpha_wave, read_line_coverage, read_segmap, read_bcg_surface_brightness, home_folder
from make_sed import get_sed
from uncover_cosmo import find_pix_per_kpc, pixel_scale

# Update to your save location
figure_save_loc = '/Users/brianlorenz/uncover/Linemaps/Figures/'

cmap='inferno'



def make_linemap(id_dr3, line_name, line_wave, line_coverage_df, supercat_df, image_size=(100,100), plot_linemaps=True):
    """Given a DR3 id and a line, make the emission line map
    
    Parameters:
    id_dr3 (int): DR3 id from UNCOVER
    line_name (str): Name of line in phot catalog, e.g. 'Halpha'
    line_coverage_df (pd.DataFrame): Corresponding line coverage datafrme, from check_line_coverage.py
    supercat_df (pd.DataFrame): from uncover_input_data
    image_size (tuple): size in pixels of the image
    plot_linemaps (boolean): Set to True to make a plot of the 3color, linemap, and contmap
    """

    line_coverage_row = line_coverage_df[line_coverage_df['id'] == id_dr3] # find the row corresponding to this object
    
    # Next, we grab all of the image files corresponding to the detected filters
    plot_3color = True # Set to false if you don't want to make the 3color plot. If you have already ran the objects before, you can set this to False
    line_filters, line_images, wht_line_images, obj_segmap, line_photfnus = make_3color(id_dr3, line_name, line_coverage_row, supercat_df, plot=plot_3color, image_size=image_size)
    # See make_3color for a description of the above

    # Gets the sedpy filter properties for the filter that the line is in
    line_sedpy_name = line_filters[1].replace('f', 'jwst_f')
    line_sedpy_filt = observate.load_filters([line_sedpy_name])[0]
    line_filter_width = line_sedpy_filt.rectangular_width
    
    line_filters_fullname = [f'f_{filt}' for filt in line_filters] # This adds back the f_ at the start. So 'f000m' becomes 'f_f000m'.

    redshift = line_coverage_row['z_50'].iloc[0] # photometric redshift from Prospector (SPS catalog)

    # Now we do all the math to get fluxes
    # This section is probably less important for the linemap work - this is just computing the total integrated line flux
    cont_percentile, line_flux, monte_carlo_lines, sed_fluxes, wave_pct, cont_value = plot_sed_around_line(id_dr3, line_name, line_wave, line_filters_fullname, redshift, line_filter_width, monte_carlo=1000)
    err_lineflux_low = np.percentile(monte_carlo_lines, 16)
    err_lineflux_high = np.percentile(monte_carlo_lines, 86)
    flux_snr = line_flux / np.std(monte_carlo_lines)
    lineflux_info = [line_flux, err_lineflux_low, err_lineflux_high, flux_snr, cont_value]

    ### Make linemaps ###
    # Need to multiply the image fluxes by 1e-8 to turn them from 10nJy to Jy
    jy_convert_factor = 1e-8
    # Get the data values in Jy
    line_red_image_data = jy_convert_factor*line_images[0].data
    line_green_image_data = jy_convert_factor*line_images[1].data
    line_blue_image_data = jy_convert_factor*line_images[2].data
    # Get the noise values in Jy
    line_red_image_noise = jy_convert_factor*(1/np.sqrt(wht_line_images[0].data))
    line_green_image_noise = jy_convert_factor*(1/np.sqrt(wht_line_images[1].data))
    line_blue_image_noise = jy_convert_factor*(1/np.sqrt(wht_line_images[2].data))
    line_image_noises = [line_red_image_noise, line_green_image_noise, line_blue_image_noise]
    
    # Math to build the linemap and propagate the uncertanties
    breakpoint()
    linemap, contmap, err_linemap = compute_line(cont_percentile, line_red_image_data, line_green_image_data, line_blue_image_data, redshift, line_filter_width, line_wave, images=True, image_noises=line_image_noises, wave_pct=wave_pct)
    

    # Compute the signal-to-noise ratio of the linemap
    linemap_snr = linemap / err_linemap
    linemap_info = [linemap, contmap, err_linemap, linemap_snr] # Will return this at the end of the funciton

    # The linemap is complete - now we make plots if enabled
    if plot_linemaps:
        # Norm values
        cont_lower_pct = 10
        cont_upper_pct = 99.99
        cont_scalea = 1e30
        linemap_lower_pct = 10
        linemap_upper_pct = 99.9
        linemap_scalea = 150

        # This is definitely not how this function was intended to be used, but this was working for me
        # Lots of room to play around with how exactly to scale and show the linemaps
        # The make_log_rgb function is from Sedona and I'm not entirely sure what it does, but it's some sort of log scaling!
        contmap_logscaled = make_log_rgb(contmap, contmap, contmap, scalea=cont_scalea)[:,:,0]
        linemap_logscaled = make_log_rgb(linemap, linemap, linemap, scalea=linemap_scalea)[:,:,0]

        # More room to play around with how to get the normalizations to make the maps pretty
        contmap_norm  = get_norm(contmap_logscaled, lower_pct=cont_lower_pct, upper_pct=cont_upper_pct)
        linemap_norm = get_norm(linemap_logscaled, lower_pct=linemap_lower_pct, upper_pct=linemap_upper_pct)

        # Setup plot - this one is a 3-panel of the 3-color, continuum, and linemaps
        fig, axarr = plt.subplots(1,3,figsize=(12,4))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        ax_image = axarr[0] 
        ax_contmap = axarr[1]
        ax_linemap = axarr[2]

        plot_single_3color(ax_image, line_images, line_filters_fullname, line_name, showtext=False)
        ax_contmap.imshow(contmap_logscaled, cmap=cmap, norm=contmap_norm) # Show the contmap with the scaling facotrs
        ax_linemap.imshow(linemap_logscaled, cmap=cmap, norm=linemap_norm) # Show the linemap with the scaling facotrs

        # Text labels and plot cleanup
        text_height = 0.92
        text_start = 0.01
        text_sep = 0.37
        ax_image.text(text_start, text_height, f'{line_filters_fullname[2][2:].upper()}', fontsize=14, transform=ax_image.transAxes, color='blue', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax_image.text(text_start+text_sep, text_height, f'{line_filters_fullname[1][2:].upper()}', fontsize=14, transform=ax_image.transAxes, color='green', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax_image.text(text_start+2*text_sep, text_height, f'{line_filters_fullname[0][2:].upper()}', fontsize=14, transform=ax_image.transAxes, color='red', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax_image.text(0.80, 0.04, f'{id_dr3}', fontsize=10, transform=ax_image.transAxes, color='white')
        axis_x = 0.05
        axis_y = 0.05
        axis_to_data = ax_image.transAxes + ax_image.transData.inverted()
        data_x, data_y = axis_to_data.transform((axis_x, axis_y))
        data_x2, data_y2 = axis_to_data.transform((axis_x, axis_y+0.02))
        ax_image.plot([data_x,data_x+(0.5/pixel_scale)], [data_y,data_y], ls='-', color='white', lw=3)
        ax_image.text(data_x, data_y2, '0.5"', color='white')

        ax_contmap.text(text_start, text_height, f'Continuum', fontsize=14, transform=ax_contmap.transAxes, color='white', path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        ax_linemap.text(text_start, text_height, f'{line_name} map', fontsize=14, transform=ax_linemap.transAxes, color='white', path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        ax_linemap.text(1-text_start, text_height, f'z={redshift:0.2f}', fontsize=14, transform=ax_linemap.transAxes, color='white', horizontalalignment='right', path_effects=[pe.withStroke(linewidth=3, foreground="black")])


        for ax in axarr:
            scale_aspect(ax)
            ax.set_xticks([]); ax.set_yticks([])

        fig.savefig(figure_save_loc+f'linemaps/{line_name}_linemaps/{id_dr3}_{line_name}_linemap.pdf', bbox_inches='tight')
        plt.close('all')
    
    return lineflux_info, linemap_info

def get_norm(image_map, scalea=1, lower_pct=10, upper_pct=99):
        """Takes a map, then gives a normalization to display it with
        Currently, it's setting the limits as percentiles of the parts of the maps that are greater than 0
        """
        imagemap_gt0 = image_map[image_map>0]
        norm = Normalize(vmin=np.percentile(imagemap_gt0,lower_pct), vmax=np.percentile(imagemap_gt0,upper_pct))
        return norm

def plot_sed_around_line(id_dr3, line_name, line_rest_wave, filters, redshift, line_filter_width, monte_carlo=1000):
    """ Gets the line flux

    Parameters:
    id_dr3 (int): DR3 from UNCOVER
    line_name (str): e.g. 'Halpha'
    line_rest_wave (float): rest wavelength of line in Angstroms
    filters (list): List of the rgb filter names in the format 'f_f000m'
    redshift (float): redshift of the object
    monte_carlo (int): Number of times to monte_carlo simulate for uncertainties
    line_filter_width (float): Filter width of the filter that contains the line (angstrom)

    Returns 
    cont_percentile (float): Percentage of the way between blue_wave and red_wave where that line is located, accounting for the higer flux continuum fitlter
    line_flux (float): Measured line flux (erg/s/cm2)
    monte_carlo_lines (array): Line fluxes from the monte carlo simulations
    sed_fluxes (list): SED fluxes observed in rgb order (Jy)
    wave_pct (float): Percentage of the way between blue_wave and red_wave where that line is located, not accounting for fluxes
    cont_value (float): Flux of the continuum (I believe in Jy)

    """
    colors = ['red', 'green', 'blue'] # For plotting

    line_wave_obs = (line_rest_wave * (1+redshift))/1e4 # micron

    sed_df = get_sed(id_dr3) # Gets the sed in an easily readable format

    fig, ax = plt.subplots(figsize=(6,6))
    ax.axvline(line_wave_obs, ls='--', color='green') # observed line

    # Plot the 3 SED points - line and two continuum filters
    # I'm sure there's a better way to loop through these but this works
    for i in range(len(filters)): # filters are in rgb order
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
    
    # Plot the SED points
    ax.errorbar(red_wave, red_flux, yerr = err_red_flux, color='red', marker='o')
    ax.errorbar(green_wave, green_flux, yerr = err_green_flux, color='green', marker='o')
    ax.errorbar(blue_wave, blue_flux, yerr = err_blue_flux, color='blue', marker='o')


    # Compute the percentile to use when combining the continuum
    connect_color = 'purple'
    wave_pct = compute_wavelength_pct(blue_wave, green_wave, red_wave)
    cont_percentile = compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux)
    
    line_flux, cont_value = compute_line(cont_percentile, red_flux, green_flux, blue_flux, redshift, 0, line_filter_width, line_rest_wave)

    # Monte carlo simulations to get uncertanties on the line flux. Probalby don't need for just line maps
    monte_carlo_lines = []
    if monte_carlo > 0:
        for i in range(monte_carlo):
            monte_carlo_red_flux = np.random.normal(loc=red_flux, scale=err_red_flux, size=1)
            monte_carlo_green_flux = np.random.normal(loc=green_flux, scale=err_green_flux, size=1)
            monte_carlo_blue_flux = np.random.normal(loc=blue_flux, scale=err_blue_flux, size=1)
            monte_carlo_cont_percentile = compute_cont_pct(blue_wave, green_wave, red_wave, monte_carlo_blue_flux, monte_carlo_red_flux)
            monte_carlo_line, monte_carlo_cont = compute_line(monte_carlo_cont_percentile, monte_carlo_red_flux[0], monte_carlo_green_flux[0], monte_carlo_blue_flux[0], redshift, line_filter_width, line_rest_wave)            
            monte_carlo_lines.append(monte_carlo_line)
    monte_carlo_lines = np.array(monte_carlo_lines)

    # Plot a few more points
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

    # Save the sed figure
    fig.savefig(figure_save_loc + f'sed_images/{line_name}_sed_images/{id_dr3}_{line_name}_sed.pdf')

    return cont_percentile, line_flux, monte_carlo_lines, sed_fluxes, wave_pct, cont_value



def make_3color(id_dr3, line_name, line_coverage_row, supercat_df, plot=False, image_size=(100,100)): 
    """Gets the image files for an object, and makes a 3 color plot

    Parameters:
    id_dr3 (int): DR3 from UNCOVER
    line_name (str): e.g. "Halpha"
    line_coverage_row (pd.DataFrame): The row in line_coverage_df corresponding to the id_dr3
    supercat_df (pd.DataFrame): from read_supercat
    plot (boolean): whether or not to make and save the plot
    image_size (tuple): in pixels, how big the image should be around the object

    Returns
    filters (list): the three filter names in rgb order
    images (list): the data of the images in rgb order. Each of these is an nxn array corresponding to image size
    wht_images (list): same as above in rgb order, but it is the uncertanties
    obj_segmap (array): segmentation map. Values are the id_DR3 values where the object is detected
    photfnus (list): in rgb order. Don't end up using these
    
    """
    obj_skycoord = get_coords(id_dr3, supercat_df) # makes an astropy "SkyCoord" object based on the ra and dec from the catalog

    # Grab the filter names, and split the name to just get 'f000m' instead of 'f_f000m'
    filt_blue = line_coverage_row[f'{line_name}_filter_bluecont'].iloc[0].split('_')[1] 
    filt_green = line_coverage_row[f'{line_name}_filter_obs'].iloc[0].split('_')[1]
    filt_red = line_coverage_row[f'{line_name}_filter_redcont'].iloc[0].split('_')[1]

    # Grouping together. I keep them in red, green, blue (rgb) order for everything even though that's not the actual wavelength order
    filters = [filt_red, filt_green, filt_blue]
    
    # Each of the next three lines takes ~0.1 seconds to run, so 0.3 seconds per object. Could re-write them to load images ahead of time to be faster if needed
    image_red, wht_image_red, photfnu_red = get_cutout(obj_skycoord, filt_red, size=image_size) # Takes the red filter and gets the cutout from the corresponding science and noise images centered on the object
    image_green, wht_image_green, photfnu_green = get_cutout(obj_skycoord, filt_green, size=image_size)
    image_blue, wht_image_blue, photfnu_blue = get_cutout(obj_skycoord, filt_blue, size=image_size)

    images = [image_red, image_green, image_blue]
    wht_images = [wht_image_red, wht_image_green, wht_image_blue]
    photfnus = [photfnu_red, photfnu_green, photfnu_blue]

    # Another spot where you can save 0.1 seconds if you have the segmap pre-loaded
    obj_segmap = get_cutout_segmap(obj_skycoord, size=image_size) # gets the cutout of the segmentation map - can use this to see the extent of the object

    if plot == True:
        fig, ax = plt.subplots(figsize = (6,6))
        plot_single_3color(ax, images, filters, line_name)
        fig.savefig(figure_save_loc + f'three_colors/{id_dr3}_{line_name}_3color.pdf')
        plt.close('all')

    return filters, images, wht_images, obj_segmap, photfnus


# Plotting single image
def plot_single_3color(ax, images, filters, line_name, showtext=True):
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
    image, wht_image, wcs, wht_wcs, photfnu = load_image(filt) # Takes about 0.1 seconds to load one of the big images
    # If you need this code faster for a huge number of objects, you could probably re-write the above to load all of the images before starting, then call the appropriate image here
    cutout = Cutout2D(image, obj_skycoord, size, wcs=wcs) # this gets the subsection of the image that centers on the galaxy with wcs magic
    wht_cutout = Cutout2D(wht_image, obj_skycoord, size, wcs=wht_wcs)
    return cutout, wht_cutout, photfnu

def get_cutout_segmap(obj_skycoord, size = (100, 100)):
    segmap, segmap_wcs = read_segmap()
    segmap_cutout = Cutout2D(segmap, obj_skycoord, size, wcs=segmap_wcs)
    return segmap_cutout

def load_image(filt):
    # Gets the science and noise image corresponding to that filter
    image_str = glob(uncover_image_folder + 'uncover_v7.*'+'*_abell2744clu_*'+filt+'*sci_f444w-matched.fits')
    wht_image_str = glob(uncover_image_folder + 'uncover_v7.*'+'*_abell2744clu_*'+filt+'*wht_f444w-matched.fits')
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

# Turns the plot into a square, mainitaining the axis limits you set
def scale_aspect(ax):
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    ydiff = np.abs(ylims[1]-ylims[0])
    xdiff = np.abs(xlims[1]-xlims[0])
    ax.set_aspect(xdiff/ydiff)

def compute_line(cont_pct, red_flx, green_flx, blue_flx, redshift, filter_width, line_rest_wave, images=False, image_noises=[], wave_pct=50):
        """
        Fluxes in Jy
        Line rest wave in angstroms
        """
        if images == True:
            cont_value = np.percentile([red_flx, blue_flx], cont_pct*100, axis=0) # If they are images, we need to combine along axis=0
        else:
            cont_value = np.percentile([blue_flx, red_flx], cont_pct*100) 

        line_value = green_flx - cont_value # Jy

        # Put in erg/s/cm2/Hz
        line_value = line_value * 1e-23
        
        # # Convert from f_nu to f_lambda
        c = 299792458 # m/s
        observed_wave = line_rest_wave * (1+redshift)
        line_value = line_value * ((c*1e10) / (observed_wave)**2) # erg/s/cm2/angstrom
 
        # Multiply by filter width to just get F
        # Filter width is observed frame width
        line_value = line_value * filter_width  # erg/s/cm2

        if images == True: # Error propagation for the images
            err_cont_value = np.sqrt((((wave_pct)**2)*(image_noises[0])**2) + (((1-wave_pct)**2)*(image_noises[2])**2))
            err_line_value = np.sqrt(image_noises[1]**2 + err_cont_value**2)
            err_line_value = err_line_value * 1e-23
            err_line_value = err_line_value * ((c*1e10) / (observed_wave)**2)
            err_line_value = err_line_value * filter_width
            return line_value, cont_value, err_line_value
        
        # For the line error propaagtion, I was using monte carlo simulations in a later part of the code
        return line_value, cont_value

def make_all_linemaps(line_name):
    redshift_sigma_threshold  = 1 # How many sigma in redshift to ensure line is in the filter
    bcg_thresh = 0.04 # Surface brightness of the bcgs across the target. I found 0.04 to be an ok threshold, but can definitely play around with this

    # loading the files
    if line_name == halpha_name:
        line_wave = halpha_wave
    supercat_df = read_supercat()
    line_coverage_df = read_line_coverage(line_name)
    bcg_df = read_bcg_surface_brightness()
    
    # This is reducing the DataFrame according to the following criteria
    line_coverage_df = line_coverage_df[line_coverage_df[f'{line_name}_redshift_sigma'] > redshift_sigma_threshold] # Reasonable redshift
    line_coverage_df = line_coverage_df[line_coverage_df['use_phot'] == 1] # use_phot 1
    line_coverage_df = line_coverage_df[line_coverage_df[f'{line_name}_all_detected'] == 1] # making sure all 3 lines are actually seen in m bands

    pandas_rows = [] # Going to save line fluxes to these - probably less relevant for the linemap projects
    for id_dr3 in line_coverage_df['id'].to_list():
        line_coverage_row = line_coverage_df[line_coverage_df['id'] == id_dr3]
        redshift_sigma = line_coverage_row[f'{line_name}_redshift_sigma'].iloc[0]
        supercat_row = supercat_df[supercat_df['id']==id_dr3]

        # Check if there are any flags for this object
        flags = []
        flags.append(supercat_row['flag_nophot'].iloc[0])
        flags.append(supercat_row['flag_lowsnr'].iloc[0])
        flags.append(supercat_row['flag_star'].iloc[0])
        flags.append(supercat_row['flag_artifact'].iloc[0])
        flags.append(supercat_row['flag_nearbcg'].iloc[0])
        if np.sum(flags) > 0: # If any flag is nonzero, skip it
            print(f'Flag found for {id_dr3}')
            continue

        # Check the bcg flag
        if bcg_df[bcg_df['id_dr3'] == id_dr3]['bcg_surface_brightness'].iloc[0] > bcg_thresh:
            print(f'Too close to bcg')
            bcg_flag = 1
            continue
        else:
            bcg_flag = 0

        # If all the flags are ok, then make the linemaps
        print(f'Making {line_name} map for {id_dr3}')
        lineflux_info, linemap_info = make_linemap(id_dr3, line_name, line_wave, line_coverage_row, supercat_df, image_size=(100,100), plot_linemaps=True)
        # Access the linemaps here - you may want to save some of these to call later in a different function
        linemap, contmap, err_linemap, err_linemap = linemap_info
        # I don't do anything else with the linemaps here since I have been focused on lineflux

        # Saving relevant lineflux info
        lineflux_info.insert(0, id_dr3)
        lineflux_info.append(redshift_sigma)
        pandas_rows.append(lineflux_info)
    # Converting lineflux info to dataframe and saving
    lineflux_df = pd.DataFrame(pandas_rows, columns=['id_dr3', f'{line_name}_flux', f'err_{line_name}_flux_low', f'err_{line_name}_flux_high', f'{line_name}_snr', f'{line_name}_cont_value', f'{line_name}_redshift_sigma'])
    lineflux_df.to_csv(f'{home_folder}/Data/lineflux_{line_name}.csv', index=False)

def make_single_halpha_linemap(id_dr3):
    # loading the files
    line_name = halpha_name
    line_wave = halpha_wave
    supercat_df = read_supercat()
    line_coverage_df = read_line_coverage(line_name)

    # make the linemap
    lineflux_info, linemap_info = make_linemap(id_dr3, line_name, line_wave, line_coverage_df, supercat_df, image_size=(100,100), plot_linemaps=True)
    
    # Can access the linemaps like this
    linemap, contmap, err_linemap, err_linemap = linemap_info
    

if __name__ == "__main__":
    # Give an id here, and this will make the linemap
    # This will be good to test individual cases, and would be a good place to try setting up a way to save the linemap
    # make_single_halpha_linemap(51980) # Takes about 2 seconds for a single map with 3 figures saved. Can save time by turning off some figures
   
    
    # This will loop through the catalog and make all linemaps according to the selection criteria specified in that function
    # You will definitely want to edit the selection criteria to match your science goals
    # And then will need to save the linemaps and determine how to best analyze them!
    make_all_linemaps(halpha_name)
    
    pass