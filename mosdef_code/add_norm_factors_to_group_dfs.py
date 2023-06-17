import initialize_mosdef_dirs as imd
from astropy.io import ascii
import numpy as np
from mosdef_obj_data_funcs import read_sed

def add_norm_factors(n_clusters):
    """Adds the normalization factors to the individual group dfs
    """

    for groupID in range(n_clusters):
        # read in the group and get the galaxies from it
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        fields = group_df['field']
        v4ids = group_df['v4id']

        norm_factors = []

        # Loop through the galaxies
        for i in range(len(group_df)):
            field = fields.iloc[i]
            v4id = v4ids.iloc[i]

            # Read in the norm_sed, which has the norm factor
            norm_sed = read_sed(field, v4id, norm=True)

            # Pull out the norm_facotr
            norm_factor = np.median(norm_sed['norm_factor'])
            norm_factors.append(norm_factor)

        # Add the column to group_df and save it
        group_df['norm_factor'] = norm_factors

        group_df.to_csv(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv', index=False)


add_norm_factors(19)