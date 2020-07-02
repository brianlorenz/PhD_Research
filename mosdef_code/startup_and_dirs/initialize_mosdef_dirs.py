from pathlib import Path
import os
import sys

home_dir = str(Path.home())
cluster_dir = home_dir + '/mosdef/Clustering'
spectra_dir = home_dir + '/mosdef/Spectra/1D'
mosdef_dir = home_dir + '/mosdef'

loc_3DHST = home_dir + '/mosdef/Surveys/3DHST/v4.1/'
loc_ZFOURGE = home_dir + '/mosdef/Surveys/ZFOURGE/'


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

    Returns:
    """
    if not os.path.exists(file_path):
        os.mkdir(file_path)
