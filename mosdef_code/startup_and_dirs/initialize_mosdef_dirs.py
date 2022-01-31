from pathlib import Path
import os
import sys

from scipy.linalg.misc import norm
from astropy.io import fits
import pandas as pd

# Set all of the directories to store files
home_dir = str(Path.home())
mosdef_dir = home_dir + '/mosdef'

cluster_dir = mosdef_dir + '/Clustering_2'
spectra_dir = mosdef_dir + '/Spectra/1D'
sed_csvs_dir = mosdef_dir + '/sed_csvs'
norm_sed_csvs_dir = mosdef_dir + '/norm_sed_csvs'
mock_sed_csvs_dir = mosdef_dir + '/mock_sed_csvs'

# Composite seds
composite_seds_dir = cluster_dir + '/composite_seds'
composite_sed_csvs_dir = composite_seds_dir + '/composite_sed_csvs'
composite_sed_images_dir = composite_seds_dir + '/composite_sed_images'
total_sed_csvs_dir = composite_seds_dir + '/total_sed_csvs'
mock_composite_sed_csvs_dir = composite_seds_dir + '/mock_composite_sed_csvs'
mock_composite_sed_images_dir = composite_seds_dir + '/mock_composite_sed_images'

# Composite filter curves
composite_filters_dir = cluster_dir + '/composite_filters'
composite_filter_csvs_dir = composite_filters_dir + '/composite_filter_csvs'
composite_filter_images_dir = composite_filters_dir + '/composite_filter_images'
composite_filter_sedpy_dir = composite_filters_dir + '/sedpy_par_files'

# Composite spectra
composite_spec_dir = cluster_dir + '/composite_spectra'

# Emission fitting
emission_fit_dir = cluster_dir + '/emission_fitting'
emission_fit_csvs_dir = emission_fit_dir + '/emission_fitting_csvs'
emission_fit_images_dir = emission_fit_dir + '/emission_fitting_images'

# Cluster plots
cluster_plot_dir = cluster_dir + '/cluster_plots'
cluster_bpt_plots_dir = cluster_plot_dir + '/bpt_diagrams'
cluster_sfr_plots_dir = cluster_plot_dir + '/mass_sfr'
cluster_uvj_plots_dir = cluster_plot_dir + '/uvj_diagrams'
cluster_similarity_plots_dir = cluster_plot_dir + '/similarities'
cluster_similarity_composite_dir = cluster_similarity_plots_dir + '/similarity_to_composite'
cluster_overview_dir = cluster_plot_dir + '/overviews'

# Prospector Outputs
prospector_output_dir = cluster_dir + '/prospector_outputs'
prospector_plot_dir = prospector_output_dir + '/prospector_plots'
prospector_h5_dir = prospector_output_dir + '/prospector_h5s'
prospector_fit_csvs_dir = prospector_output_dir + '/prospector_csvs'
prospector_emission_fits_dir = prospector_output_dir + '/prospector_emission_fits'


# Composite seds with spectra merged
composite_seds_spec_dir = cluster_dir + '/composite_sed_and_spec'
composite_seds_spec_csvs_dir = composite_seds_spec_dir + '/composite_sed_and_spec_csvs'
composite_seds_spec_images_dir = composite_seds_spec_dir + '/composite_sed_and_spec_images'

# Line widths
line_widths_dir = cluster_dir + '/line_widths'
line_width_csvs_dir = line_widths_dir + '/line_width_csvs'
line_width_images_dir = line_widths_dir + '/line_width_images'


# Folders to store UVJ measurements
uvj_dir = mosdef_dir + '/UVJ_Colors'
composite_uvj_dir = cluster_dir + '/UVJ_Colors_composite'

# Axis ratios
loc_axis_ratio_cat = mosdef_dir + '/axis_ratio_data/Merged_catalogs/mosdef_all_cats_v2.csv'
axis_output_dir = mosdef_dir + '/axis_ratio_outputs'
axis_figure_dir = axis_output_dir + '/axis_ratio_figures'

axis_cluster_data_dir = mosdef_dir + '/axis_ratio_data_clustered'

# Folder for coutns at each step
gal_counts_dir = mosdef_dir + '/galaxy_counts'

#Folder for where to generate paper figures
paper_fig_dir = mosdef_dir + '/paper_figures'


# Misc mosdef files
loc_3DHST = home_dir + '/mosdef/Surveys/3DHST/v4.1/'
loc_ZFOURGE = home_dir + '/mosdef/Surveys/ZFOURGE/'
loc_linemeas = mosdef_dir + '/Mosdef_cats/linemeas_latest.csv'
loc_uvj = mosdef_dir + '/Mosdef_cats/uvj_latest.dat'
loc_mosdef0d = mosdef_dir + '/Mosdef_cats/mosdef_0d_latest.fits'
loc_sfrs_latest = mosdef_dir + '/Mosdef_cats/mosdef_sfrs_latest.fits'
loc_agnflag_latest = mosdef_dir + '/Mosdef_cats/agnflag_latest.fits'
loc_mosdef_elines = mosdef_dir + '/Catalogs/Nebular_Emission/mosdef_elines.txt' # This is a file that I generated
median_zs_file = composite_seds_dir + '/median_zs.csv' # Also generated
number_agn_file = cluster_dir + '/number_agn.csv'
bad_groups_file = cluster_dir + '/bad_groups.csv'
loc_galaxy_uvjs = mosdef_dir + '/UVJ_Colors/galaxy_uvjs.csv'
loc_eazy_uvj_fits = mosdef_dir + '/Mosdef_cats/mosdef_eazy_uvj_latest.fits'
loc_eazy_uvj_cat = mosdef_dir + '/Mosdef_cats/mosdef_eazy_uvj_latest.csv'
loc_eqwidth_cat = mosdef_dir + '/Mosdef_cats/compile_ew_bc03smc0p2.txt'



# FAST outputs
FAST_dir = mosdef_dir + '/FAST'





# Filters
# Location of the translate filter file for mosdef
mosdef_filter_translate = home_dir + '/code/mosdef_code/filters/catalog_filters/FILTER.RES.latest'
mosdef_filter_overview = home_dir + '/code/mosdef_code/filters/catalog_filters/overview'


all_dirs = [prospector_emission_fits_dir, composite_seds_spec_dir, composite_seds_spec_csvs_dir, composite_seds_spec_images_dir, prospector_plot_dir, prospector_fit_csvs_dir, composite_filter_sedpy_dir, cluster_plot_dir, mock_composite_sed_images_dir, mock_composite_sed_csvs_dir, cluster_similarity_plots_dir, cluster_similarity_composite_dir, composite_uvj_dir, uvj_dir, cluster_plot_dir, cluster_uvj_plots_dir, cluster_bpt_plots_dir, cluster_sfr_plots_dir, emission_fit_dir, emission_fit_csvs_dir, emission_fit_images_dir, composite_spec_dir, composite_seds_dir, composite_filters_dir, composite_filter_images_dir, composite_filter_csvs_dir, composite_sed_images_dir, composite_sed_csvs_dir, cluster_dir, spectra_dir, sed_csvs_dir, norm_sed_csvs_dir, mock_sed_csvs_dir, composite_seds_dir]

def setup_cluster_dirs(all_dirs):
    for dir in all_dirs:
        check_and_make_dir(dir)


def reset_cluster_dirs(cluster_dir):
    """Sets up the file structure for the Clustering directory if it has just been cleaned

    Parameters:
    cluster_dir (string): The 'Clustering' director to populate

    Returns:
    """

    check_and_make_dir(cluster_dir)
    check_and_make_dir(cluster_dir + '/cluster_stats')
    check_and_make_dir(cluster_dir + '/cluster_stats/similarities')
    check_and_make_dir(
        cluster_dir + '/cluster_stats/similarities/similarities_composite')
    check_and_make_dir(cluster_dir + '/cluster_stats/uvj_diagrams')
    check_and_make_dir(cluster_dir + '/composite_filter')
    check_and_make_dir(cluster_dir + '/composite_seds')
    check_and_make_dir(cluster_dir + '/composite_seds')
    check_and_make_dir(cluster_dir + '/composite_seds/composite_seds_nofilt')
    check_and_make_dir(cluster_dir + '/composite_spectra')
    check_and_make_dir(cluster_dir + '/composite_spectra/cluster_norm')
    check_and_make_dir(cluster_dir + '/composite_spectra/composite_sed_norm')
    check_and_make_dir(cluster_dir + '/emission_fitting')


def reset_sed_dirs(mosdef_dir):
    """Sets up the file structure for the Clustering directory if it has just been cleaned

    Parameters:
    cluster_dir (string): The 'Clustering' director to populate

    Returns:
    """
    check_and_make_dir(cluster_dir + '/SED_Images')
    check_and_make_dir(cluster_dir + '/SED_Images/mock_sed_images')
    check_and_make_dir(cluster_dir + '/SED_Images/mock_composite_sed_images')
    check_and_make_dir(cluster_dir + '/mock_sed_csvs')
    check_and_make_dir(cluster_dir + '/mock_sed_csvs/mock_composite_sed_csvs')


def check_and_make_dir(file_path):
    """Checks to see if a directory exists - if not, creates the directory

    Parameters:
    file_path (str): Path to a directory that you wish to create

    Returns:
    """
    if not os.path.exists(file_path):
        os.mkdir(file_path)


def setup_eazy_uvj_cat():
    uvj_cat = fits.open(loc_eazy_uvj_fits)[1].data
    uvj_cat_df = pd.DataFrame(uvj_cat)
    uvj_cat_df = uvj_cat_df.rename(columns={"VJ": "V_J", "UV": "U_V"}) 
    uvj_cat_df.to_csv(loc_eazy_uvj_cat, index=False)



# setup_cluster_dirs(all_dirs)
