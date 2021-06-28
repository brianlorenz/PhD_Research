from pathlib import Path
import os
import sys

from scipy.linalg.misc import norm

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

# Composite filter curves
composite_filters_dir = cluster_dir + '/composite_filters'
composite_filter_csvs_dir = composite_filters_dir + '/composite_filter_csvs'
composite_filter_images_dir = composite_filters_dir + '/composite_filter_images'

# Composite spectra
composite_spec_dir = cluster_dir + '/composite_spectra'

# Emission fitting
emission_fit_dir = cluster_dir + '/emission_fitting'
emission_fit_csvs_dir = emission_fit_dir + '/emission_fitting_csvs'
emission_fit_images_dir = emission_fit_dir + '/emission_fitting_images'

# Cluster plots
cluster_plot_dir = cluster_dir + '/cluster_plots'
cluster_bpt_plots_dir = cluster_plot_dir + '/bpt_diagrams'

# Misc mosdef files
loc_3DHST = home_dir + '/mosdef/Surveys/3DHST/v4.1/'
loc_ZFOURGE = home_dir + '/mosdef/Surveys/ZFOURGE/'
loc_linemeas = mosdef_dir + '/Mosdef_cats/linemeas_latest.csv'

# Filters
# Location of the translate filter file for mosdef
mosdef_filter_translate = home_dir + '/code/mosdef_code/filters/catalog_filters/FILTER.RES.latest'
mosdef_filter_overview = home_dir + '/code/mosdef_code/filters/catalog_filters/overview'


all_dirs = [emission_fit_dir, emission_fit_csvs_dir, emission_fit_images_dir, composite_spec_dir, composite_seds_dir, composite_filters_dir, composite_filter_images_dir, composite_filter_csvs_dir, composite_sed_images_dir, composite_sed_csvs_dir, cluster_dir, spectra_dir, sed_csvs_dir, norm_sed_csvs_dir, mock_sed_csvs_dir, composite_seds_dir]

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

# setup_cluster_dirs(all_dirs)