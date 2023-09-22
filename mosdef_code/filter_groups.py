# Writes a text file with a list of groups that have too few galaxies or unusable data

from astropy.io import ascii
import pandas as pd
import initialize_mosdef_dirs as imd
from cluster_stats import plot_all_similarity


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

# remove_groups_by_similiary(23)
# generate_skip_file(23)