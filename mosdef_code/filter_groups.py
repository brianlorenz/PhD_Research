
from astropy.io import ascii
import pandas as pd
import initialize_mosdef_dirs as imd
from cluster_stats import plot_all_similarity
from mosdef_obj_data_funcs import read_sed
import numpy as np

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
    for i in range(len(zobjs)):
        field = zobjs.iloc[i]['field']
        v4id = zobjs.iloc[i]['v4id']
        cluster_num = zobjs.iloc[i]['cluster_num']
        sed = read_sed(field, v4id)

        # remove if: Flux between 2500-10000 angstroms has a negative value that is not -99
        wave_range = [2500, 10000]
        sed_idxs = np.logical_and(sed['rest_wavelength']>wave_range[0], sed['rest_wavelength']<wave_range[1])
        bad_filts = np.logical_and(sed[sed_idxs]['f_lambda'] > -90, sed[sed_idxs]['f_lambda'] < 0)
        if np.sum(bad_filts) > 0:
            zobjs.loc[i, 'flag_sed'] = 1
    print(zobjs[zobjs['flag_sed'] > 0])
    zobjs.to_csv(imd.cluster_dir+'/zobjs_clustered.csv', index=False)

# find_bad_seds(20)
# remove_groups_by_similiary(23)
# generate_skip_file(23)