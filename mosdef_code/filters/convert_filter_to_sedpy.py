# run convert_filter_to_sedpy.py
# '/Users/brianlorenz/mosdef/composite_sed_csvs/composite_filter_csvs/0_filter_csvs/'

import os
import numpy as np
from astropy.io import ascii
import pandas as pd
import initialize_mosdef_dirs as imd
from prospector_composite_params import to_median_redshift


# e.g. target_folder =
# '/Users/brianlorenz/mosdef/composite_sed_csvs/composite_filter_csvs/0_filter_csvs/'


def find_median_redshifts(n_clusters):
    """Finds the median redshift within a composite sed group

    Parameters:
    n_clusters (int): total number of clusters

    Returns:
    median_z (float): median redshift in that group

    """
    median_zs = []
    groupIDs = []
    for groupID in range(n_clusters):
        # files_list = os.listdir(imd.cluster_dir + f'/{groupID}')
        # names_list = [file[:-9] for file in files_list]
        # z_list = []
        # for name in names_list:
        #     try:
        #         z = ascii.read(
        #             f'/Users/brianlorenz/mosdef/sed_csvs/{name}_sed.csv').to_pandas()['Z_MOSFIRE'].iloc[0]
        #     except FileNotFoundError:
        #         z = ascii.read(
        #             f'/Users/brianlorenz/mosdef/sed_csvs/{name}_3DHST_sed.csv').to_pandas()['Z_MOSFIRE'].iloc[0]
        #     z_list.append(z)
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        breakpoint()
        median_z = np.median(z_list)
        median_zs.append(median_z)
        groupIDs.append(groupID)

    
    median_z_df = pd.DataFrame(zip(groupIDs, median_zs), columns=[
                               'groupID', 'median_z'])
    median_z_df.to_csv(imd.composite_seds_dir + '/median_zs.csv', index=False)
    return


def de_median_redshift(wavelength, median_z):
    """De-redshifts the same way as above

    Parameters:
    wavelength (array): wavelength values to convert
    median_z (int): redshift to change the wavelength by

    Returns:
    wavelength_red (array): redshifted wavelength

    """

    wavelength_red = wavelength / (1 + median_z)
    return wavelength_red


def convert_filters_to_sedpy(target_folder, groupID):
    """Converts every filter csv in the target folder into a style readable by sedpy

    Parameters:
    target_folder (str): location of folder containing filter points
    groupID (int): Id of the group corresponding to that folder

    """

    # Figure out which files  in the folder are csv filters:
    filt_files = [file for file in os.listdir(target_folder) if '.csv' in file]

    # Loop through files one at a time
    for file in filt_files:
        print(f'    Converting {file}')
        # Read the wavelength/transmisison data
        data = ascii.read(target_folder + file).to_pandas()
        # Find the range that we care about, only the nonzero values
        nonzero_points = np.array(data[data['transmission'] > 0].index)
        # Add 2 points on either side to serve as zeroes
        prepend_val = nonzero_points[0] - 2
        append_val = nonzero_points[-1] + 2
        if prepend_val >= 0:
            nonzero_points = np.insert(
                nonzero_points, 0, [prepend_val, prepend_val + 1])
        if append_val <= data.index[-1]:
            nonzero_points = np.append(
                nonzero_points, [append_val - 1, append_val])

        # Append zeroes to the file number if they are less than 5 long
        new_name = append_zeros_to_filtname(file)

        data = data.iloc[nonzero_points]
        # This would save it as-is, but we have determined that it needs to be shifted to the median redshift of the group
        # data.to_csv(target_folder + new_name.replace('.csv', '.par'),
        #             index=False, sep=' ', header=False)

        # Redshift to the medain redshift of the group
        import initialize_mosdef_dirs as imd
        zs_df = ascii.read(imd.median_zs_file).to_pandas()
        median_z = zs_df[zs_df['groupID'] == groupID]['median_z'].iloc[0]
        data['rest_wavelength'] = to_median_redshift(
            data['rest_wavelength'], median_z)
        save_folder = imd.composite_filter_sedpy_dir + f'/{groupID}_sedpy_pars'
        imd.check_and_make_dir(save_folder)
        data.to_csv(save_folder + '/' + new_name.replace('.csv', '_red.par'),
                    index=False, sep=' ', header=False)


def convert_all_folders_to_sedpy(n_clusters):
    """Runs the above convert_folder_to_sedpy script on multiple folders

    Parameters:
        n_clusters(int): Number of clusters
    """
    for groupID in range(n_clusters):
        target_folder = imd.composite_filter_csvs_dir + \
            f'/{groupID}_filter_csvs/'
        print(
            f'Converting {target_folder} to sedpy format, for group {groupID}')
        convert_filters_to_sedpy(target_folder, groupID)


def append_zeros_to_filtname(filtname):
    """Adds zeros to standardize the size of all filternames

    Parameters:
    filtname (str) - name of the filter file

    Returns:
    filtname (str) - name of the filter file, possibly now with zeroes inserted

    """
    while len(filtname) < 15:
        filtname = filtname[:6] + '0' + filtname[6:]
    return filtname

