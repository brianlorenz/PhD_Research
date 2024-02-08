
from astropy.io import ascii
import pandas as pd
import initialize_mosdef_dirs as imd
from cluster_stats import plot_all_similarity
from mosdef_obj_data_funcs import read_sed, read_mock_sed
from cross_correlate import get_cross_cor
import numpy as np
from scipy import interpolate


# def generate_skip_file(n_groups):
#     """Makes the file that removes groups until you have 20 (savio node size)
    
#     """
#     n_agn_df = ascii.read(imd.number_agn_file).to_pandas()
#     sorted_agn_df = n_agn_df.sort_values('n_gals')
#     bad_groups_df =  sorted_agn_df.iloc[:-20]['groupID']
#     bad_groups_df.to_csv(imd.bad_groups_file, index=False)

def remove_groups_by_similiary(n_groups, sim_thresh=0.8):
    """Fills the skip file with groups that have similarities lower than the threshold
    
    """
    plot_all_similarity(n_groups)
    similarity_df = ascii.read(imd.cluster_similarity_plots_dir+'/composite_similarities.csv').to_pandas()
    similarity_df = similarity_df[similarity_df['mean_sim']<sim_thresh]
    bad_groups = similarity_df['groupID']
    bad_groups.to_csv(imd.bad_groups_file, index=False)

def find_bad_seds(n_groups):
    """Lists galaxies form their groups with bad SED data
    
    """
    zobjs = ascii.read(imd.cluster_dir+'/zobjs_clustered.csv').to_pandas()
    zobjs['flag_sed'] = np.zeros(len(zobjs))
    zobjs['sed_corr'] = np.zeros(len(zobjs))
    
    for i in range(len(zobjs)):
        field = zobjs.iloc[i]['field']
        v4id = zobjs.iloc[i]['v4id']
        cluster_num = zobjs.iloc[i]['cluster_num']
        sed = read_sed(field, v4id)
        mock_sed = read_mock_sed(field, v4id)
        interp_mock = interpolate.interp1d(mock_sed['rest_wavelength'], mock_sed['f_lambda'], fill_value=-99, bounds_error=False)
        sed_new = sed.copy(deep=True)
        sed_new['f_lambda'] = interp_mock(sed['peak_wavelength'])
        def get_variance(mock_sed_1, mock_sed_2):
            f1 = mock_sed_1['f_lambda']
            f2 = mock_sed_2['f_lambda']

            f1_good = f1 > -10
            f2_good = f2 > -10
            both_good = np.logical_and(f1_good, f2_good)

            # Filter down to only the indicies where both SEDs have real values
            f1 = f1[both_good]
            f2 = f2[both_good]
            for i in range(len(f1)):
                variance = np.sum((f1.iloc[i]-f2.iloc[i])**2)
            return variance
        variance = get_variance(sed, sed_new)
        zobjs.loc[i, 'sed_corr'] = variance
        # filt_row = sed['filter_name']=='f_K'
        # filt_sn = sed[filt_row]['rest_f_lambda']/sed[filt_row]['rest_err_f_lambda']
        # remove if: Flux between 2500-10000 angstroms has a negative value that is not -99
        # wave_range = [2500, 10000]
        # sed_idxs = np.logical_and(sed['rest_wavelength']>wave_range[0], sed['rest_wavelength']<wave_range[1])
        # bad_filts = np.logical_and(sed[sed_idxs]['f_lambda'] > -90, sed[sed_idxs]['f_lambda'] < 0)
        # if np.sum(bad_filts) > 0:
        #     zobjs.loc[i, 'flag_sed'] = 1
    breakpoint()
    print(zobjs[zobjs['flag_sed'] > 0])
    # zobjs.to_csv(imd.cluster_dir+'/zobjs_clustered.csv', index=False)

find_bad_seds(20)
# remove_groups_by_similiary(23)
# generate_skip_file(23)